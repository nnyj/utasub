"""Global monotonic CTC forced alignment over romaji (torchaudio MMS_FA).

One Viterbi path over the whole audio span, seedless and unwindowed: every
line's tokens in order, with a `<star>` wildcard at head, tail, and between
lines so intros, solos, MC talk have a label to sit on. Stamps remain in
order, but misplaced stamps still need warp filtering.

Confidence is mean token log-prob of a line's own tokens. Line ends use the
per-song GATE_PCT percentile; start adoption uses a timing bound. Warp fit
gets the ungated set, since its own chain filters discard outliers.

Absence of torchaudio degrades with a message, never crashes.
"""
import statistics

SR = 16000
# emission is computed in chunks (self-attention is quadratic in frames) with
# EMIT_CTX of receptive-field context trimmed off each seam; the alignment path
# stays one forced_align over the concatenated emission
EMIT_CHUNK, EMIT_CTX = 30.0, 1.5
GATE_PCT = 10  # drop this % of line ends by mean token log-prob

_model_cache = []
_available = None

# ━━━━━━ model and emission ━━━━━━

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
  """[(line_idx, [token ids], [romaji chars])], one entry per line that
  romanizes to anything. The char list is the same sequence as the ids, so a
  token span maps back to the romaji character it timed."""
  from .romanize import detect, get_default_locale, romanize
  locale = get_default_locale() or detect(" ".join(texts))
  out = []
  for j, text in enumerate(texts):
    rom = romanize(text, locale=locale) or text
    # '-' is the blank label in the MMS dict, never a target
    chars = [c for w in rom.lower().split() for c in w
             if c in dictionary and c != "-"]
    if chars:
      out.append((j, [dictionary[c] for c in chars], chars))
  return out

# ━━━━━━ align ━━━━━━

def align_lines(texts, audio):
  """Stamp every line against 16k audio in one global pass.
  Returns (starts, ends, scores, token_spans) keyed by line index, scores =
  mean token log-prob, token_spans = [(romaji char, start, end)] for karaoke
  fills. Empty dicts when the pass cannot run."""
  model, dictionary, torch, dev = load_model()
  import torchaudio.functional as AF
  star = dictionary["<star>"]
  lines = _line_tokens(texts, dictionary)
  if not lines or len(audio) < SR // 2:
    return {}, {}, {}, {}
  em, sec_per_frame = _emission(model, audio, torch, dev)
  seq, slices = [star], []
  for j, ids, chars in lines:
    slices.append((j, len(seq), len(seq) + len(ids), chars))
    seq += ids + [star]
  if len(seq) > em.shape[1]:  # more tokens than frames: nothing alignable
    print(f"  ctc: {len(seq)} tokens over {em.shape[1]} frames, skipped")
    return {}, {}, {}, {}
  targets = torch.tensor([seq], dtype=torch.int32, device=dev)
  aligned, path_scores = AF.forced_align(em, targets, blank=0)
  spans = AF.merge_tokens(aligned[0], path_scores[0])
  if len(spans) != len(seq):
    print(f"  ctc: {len(spans)} spans for {len(seq)} tokens, skipped")
    return {}, {}, {}, {}
  starts, ends, scores, token_spans = {}, {}, {}, {}
  for j, a, b, chars in slices:
    starts[j] = spans[a].start * sec_per_frame
    ends[j] = spans[b - 1].end * sec_per_frame
    scores[j] = statistics.mean(s.score for s in spans[a:b])
    token_spans[j] = [(c, s.start * sec_per_frame, s.end * sec_per_frame)
                      for c, s in zip(chars, spans[a:b])]
  return starts, ends, scores, token_spans

# ━━━━━━ confidence gate ━━━━━━

def _pct_cut(vals, pct):
  """Value at the bottom pct% of sorted vals."""
  return vals[min(len(vals) - 1, int(pct / 100.0 * len(vals)))]

def low_conf_cut(scores, pct=GATE_PCT):
  """Score below which a cue is worth a second look, in mean-token-log-prob:
  the bottom pct% of this song. Relative only: MMS_FA is a speech model, so a
  loud sung track scores every line low and an absolute floor would flag all of
  it. None with fewer than 5 scored cues, too few to rank. `scores` is any
  iterable, None ignored."""
  vals = sorted(s for s in scores if s is not None)
  if len(vals) < 5:
    return None
  return _pct_cut(vals, pct)

def gate(stamps, scores, pct=GATE_PCT):
  """Drop the lowest pct% of lines by confidence. One per-song percentile,
  constant for the whole cut, so a line's fate doesn't depend on which
  neighbours happen to be around it."""
  if not stamps or pct <= 0 or not scores:
    return dict(stamps)
  vals = sorted(scores[j] for j in stamps if j in scores)
  if not vals:
    return dict(stamps)
  cut = _pct_cut(vals, pct)
  return {j: t for j, t in stamps.items() if scores.get(j, cut) >= cut}
