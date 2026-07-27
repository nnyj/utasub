"""Global monotonic CTC forced alignment over romaji (torchaudio MMS_FA).

One Viterbi path over the whole audio span, seedless and unwindowed: every
line's tokens in order, with a `<star>` wildcard at head, tail, and between
lines so intros, solos, MC talk have a label to sit on and no lyric token
gets dragged onto them. Lines are therefore in order and on the right
occurrence of a repeated chorus by construction.

Confidence is mean token log-prob of a line's own tokens; separates outliers
cleanly, so stamps consumed directly (place._adopt_fa) are gated at the
per-song GATE_PCT percentile. Warp fit gets the ungated set, since its own
chain filters already discard outliers.

Absence of torchaudio degrades with a message, never crashes.
"""
import statistics

SR = 16000
# emission is computed in chunks (self-attention is quadratic in frames) with
# EMIT_CTX of receptive-field context trimmed off each seam; the alignment path
# stays one forced_align over the concatenated emission
EMIT_CHUNK, EMIT_CTX = 30.0, 1.5
GATE_PCT = 10  # drop this % of lines by mean token log-prob before direct use

_model_cache = []
_available = None


def ctc_available():
  """True when torch + torchaudio (with MMS_FA) are importable."""
  global _available
  if _available is None:
    try:
      import torch  # noqa: F401
      import torchaudio
      _available = hasattr(torchaudio.pipelines, "MMS_FA")
    except ImportError:
      _available = False
  return _available


def load_model():
  """Lazy MMS_FA bundle with the star label. (model, dict, torch, device)."""
  if not _model_cache:
    import torch
    import torchaudio
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.MMS_FA
    print("  loading MMS_FA aligner...")
    _model_cache.append((bundle.get_model(with_star=True).eval().to(dev),
                         bundle.get_dict(star="<star>"), torch, dev))
  return _model_cache[0]


def _emission(model, audio, torch, dev):
  """(log-prob emission over the whole span, seconds per frame).

  Simplification: seam frame counts are rounded, so a chunk boundary can shift
  a frame (20ms) against a single-pass emission. Upgrade path: cut on exact
  stride multiples."""
  step, ctx = int(EMIT_CHUNK * SR), int(EMIT_CTX * SR)
  parts = []
  for a in range(0, len(audio), step):
    b = min(len(audio), a + step)
    lo, hi = max(0, a - ctx), min(len(audio), b + ctx)
    wav = torch.from_numpy(audio[lo:hi].copy()).unsqueeze(0).to(dev)
    with torch.inference_mode():
      emission, _ = model(wav)
    per_frame = (hi - lo) / emission.shape[1]
    d0 = round((a - lo) / per_frame)
    d1 = emission.shape[1] - round((hi - b) / per_frame)
    parts.append(emission[0, d0:d1].float())
  em = torch.cat(parts).unsqueeze(0)
  return em, len(audio) / em.shape[1] / SR


def _line_tokens(texts, dictionary):
  """[(line_idx, [token ids])], one entry per line that romanizes to anything."""
  from .romanize import detect, get_default_locale, romanize
  locale = get_default_locale() or detect(" ".join(texts))
  out = []
  for j, text in enumerate(texts):
    rom = romanize(text, locale=locale) or text
    # '-' is the blank label in the MMS dict, never a target
    ids = [dictionary[c] for w in rom.lower().split() for c in w
           if c in dictionary and c != "-"]
    if ids:
      out.append((j, ids))
  return out


def align_lines(texts, audio, sr=SR):
  """Stamp every line against audio in one global pass.
  Returns (starts, ends, scores) keyed by line index, scores = mean token
  log-prob. Empty dicts when the pass cannot run."""
  model, dictionary, torch, dev = load_model()
  import torchaudio.functional as AF
  star = dictionary["<star>"]
  lines = _line_tokens(texts, dictionary)
  if not lines or len(audio) < sr // 2:
    return {}, {}, {}
  em, sec_per_frame = _emission(model, audio, torch, dev)
  seq, slices = [star], []
  for j, ids in lines:
    slices.append((j, len(seq), len(seq) + len(ids)))
    seq += ids + [star]
  if len(seq) > em.shape[1]:  # more tokens than frames: nothing alignable
    print(f"  ctc: {len(seq)} tokens over {em.shape[1]} frames, skipped")
    return {}, {}, {}
  targets = torch.tensor([seq], dtype=torch.int32, device=dev)
  aligned, path_scores = AF.forced_align(em, targets, blank=0)
  spans = AF.merge_tokens(aligned[0], path_scores[0])
  if len(spans) != len(seq):
    print(f"  ctc: {len(spans)} spans for {len(seq)} tokens, skipped")
    return {}, {}, {}
  starts, ends, scores = {}, {}, {}
  for j, a, b in slices:
    starts[j] = spans[a].start * sec_per_frame
    ends[j] = spans[b - 1].end * sec_per_frame
    scores[j] = statistics.mean(s.score for s in spans[a:b])
  return starts, ends, scores


def gate(stamps, scores, pct=GATE_PCT):
  """Drop the lowest pct% of lines by confidence. One per-song percentile,
  constant for the whole cut, so a line's fate doesn't depend on which
  neighbours happen to be around it."""
  if not stamps or pct <= 0 or not scores:
    return dict(stamps)
  vals = sorted(scores[j] for j in stamps if j in scores)
  if not vals:
    return dict(stamps)
  cut = vals[min(len(vals) - 1, int(pct / 100.0 * len(vals)))]
  return {j: t for j, t in stamps.items() if scores.get(j, cut) >= cut}
