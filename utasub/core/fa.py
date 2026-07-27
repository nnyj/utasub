"""Forced alignment via Qwen3-ForcedAligner-0.6B. Stamps known lyric text against
audio per section, refining coarse line starts. Absence of qwen_asr degrades with
message, never crashes."""
import re
import unicodedata

from .romanize import detect, get_default_locale

_aligner_cache = []
_available = None
# aligner takes whisper-style language names; '' (latin) -> English
_FA_LANG = {"ja": "Japanese", "zh": "Chinese", "ko": "Korean"}


def fa_available():
  """True when qwen_asr is importable."""
  global _available
  if _available is None:
    try:
      import qwen_asr  # noqa: F401
      _available = True
    except ImportError:
      _available = False
  return _available


def load_aligner():
  """Lazy Qwen3-ForcedAligner (0.6B), cuda bfloat16 when available."""
  if not _aligner_cache:
    import torch
    from qwen_asr import Qwen3ForcedAligner
    # determinism across sessions (CUBLAS_WORKSPACE_CONFIG set in package
    # __init__): without it stamps drift with GPU/machine state. warn_only
    # since some kernels have no deterministic implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print("  loading Qwen3-ForcedAligner-0.6B...")
    _aligner_cache.append(Qwen3ForcedAligner.from_pretrained(
      "Qwen/Qwen3-ForcedAligner-0.6B", dtype=dtype, device_map=device))
  return _aligner_cache[0]


def _norm_key(t):
  """Lowercase NFKC, keep digits+ascii+kana+kanji only."""
  return re.sub(r"[^0-9a-z぀-ヿ一-鿿]", "",
                unicodedata.normalize("NFKC", t).lower())


def _split_sections(starts, gap=4.0, max_span=150.0):
  """Group line indices into sections split at start gaps > gap or span > max_span."""
  secs = []
  cur = [0]
  for j in range(1, len(starts)):
    if starts[j] - starts[j - 1] > gap or starts[j] - starts[cur[0]] > max_span:
      secs.append(cur)
      cur = []
    cur.append(j)
  secs.append(cur)
  return secs


def force_align_lines(starts, texts, audio, sr=16000, pre_pad=5.0, post_pad=10.0,
                      drift_gate=8.0):
  """Refine line times by force-aligning lyric text over per-section audio windows.
  Coarse placement picks windows, aligner nails lines. Token-to-line mapping by
  cumulative normalized-char position. Singable-rate plausibility gate <25 chars/s.
  pre_pad/post_pad size the section windows; drift_gate=None lets a stamp land
  arbitrarily far from its seed (the warp fitter judges that itself).
  Returns ({line_idx: start}, {line_idx: end})."""
  aligner = load_aligner()
  secs = _split_sections(starts)
  batch_audio, batch_text, batch_lang, metas = [], [], [], []
  dur = len(audio) / sr
  L = len(starts)
  song_locale = get_default_locale()  # song-level pin, per-section detect fallback
  # precompute norm keys per line (avoids recompute on context-overlap lines)
  norm_keys = [_norm_key(texts[j]) for j in range(L)]

  for sec in secs:
    # context lines: neighbors whose voice must be in aligned text to avoid edge drift
    ctx = [sec[0] - 1] if sec[0] > 0 else []
    post = [sec[-1] + 1] if sec[-1] + 1 < L else []
    w0 = max(0.0, (starts[ctx[0]] if ctx else starts[sec[0]] - 2.5) - pre_pad)
    w1 = min(dur, (starts[post[0]] + post_pad * 0.4) if post
             else starts[sec[-1]] + post_pad)
    seq = ctx + sec + post
    sec_text = " ".join(texts[j] for j in seq)
    clip = audio[int(w0 * sr):int(w1 * sr)]
    if len(clip) < sr // 10:  # seed landed off the end: nothing to align against
      continue
    batch_audio.append((clip, sr))
    batch_text.append(sec_text)
    batch_lang.append(_FA_LANG.get(song_locale or detect(sec_text), "English"))
    metas.append((seq, set(sec), w0))

  new_s, new_e = {}, {}
  results = aligner.align(audio=batch_audio, text=batch_text, language=batch_lang)
  for (seq, want, w0), res in zip(metas, results):
    keys = [norm_keys[j] for j in seq]
    bounds = [0]
    for k in keys:
      bounds.append(bounds[-1] + len(k))
    acc = 0
    li = 0
    got_s, got_e, got_e_raw = {}, {}, {}
    for it in res.items:
      tl = len(_norm_key(it.text))
      if not tl:
        continue
      while li + 1 < len(seq) and bounds[li + 1] <= acc:
        li += 1
      # collapsed tokens (zero-duration, piled at one spot) = aligner failure;
      # skip got_s but still advance acc for boundary tracking
      collapsed = abs(it.end_time - it.start_time) < 0.001
      if not collapsed:
        for k in range(li, len(seq)):
          if bounds[k] >= acc + tl:
            break
          if bounds[k] >= acc and seq[k] not in got_s:
            got_s[seq[k]] = w0 + it.start_time
        got_e_raw[seq[li]] = w0 + it.end_time
        # a token whose chars cross into the next line ends on that line's
        # audio, so its end time isn't this line's end; the raw copy still
        # feeds the rate gate. Dropping is safe: build_cues takes
        # max(envelope, hint), only a too-long hint can hurt.
        if li + 1 >= len(seq) or acc + tl <= bounds[li + 1]:
          got_e[seq[li]] = w0 + it.end_time
      acc += tl
    if acc != bounds[-1]:  # token mismatch: keep coarse
      continue
    # plausibility gates:
    # 1. singable rate <25 norm chars/s (collapsed stamps)
    # 2. coarse drift <4s (wrong-occurrence matches on repeated lyrics)
    js = [j for j in seq if j in got_s]
    for i, j in enumerate(js):
      if j not in want:
        continue
      end = got_e_raw.get(j)
      if end is None:
        end = next((got_s[k] for k in js[i + 1:] if got_s[k] > got_s[j]), got_s[j])
      chars = len(norm_keys[j])
      if chars >= 3 and chars > max(end - got_s[j], 0.0) * 25:
        continue
      if drift_gate is not None and abs(got_s[j] - starts[j]) > drift_gate:
        continue
      new_s[j] = got_s[j]
      if j in got_e:
        new_e[j] = got_e[j]
  return new_s, new_e
