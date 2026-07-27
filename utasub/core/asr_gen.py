"""Generate the ASR block in <name>.utasub.json from media via Qwen3-ASR
(optional [asr] extra).

Core stays load-only; this is the one module running an ASR model, only when
the user asks (Transcribe button / `utasub-asr` CLI). Heavy deps (qwen-asr,
audio-separator, torch) import lazily inside functions.

Writes "asr" block ({"language": str, "segments": [[s, e, t], ...]}) via
session.save_asr."""
import os
import re
import time
from pathlib import Path

from . import session
from .align import find_vocals, load_audio_16k, stem_cache_dir
from .romanize import is_cjk

LANG_CODES_TO_NAMES = {
  "ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean",
  "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
  "ru": "Russian", "it": "Italian", "ar": "Arabic", "th": "Thai",
  "vi": "Vietnamese", "tr": "Turkish", "hi": "Hindi", "id": "Indonesian",
  "ms": "Malay", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
  "fi": "Finnish", "pl": "Polish", "cs": "Czech",
}

DEFAULT_STEM_MODEL = "vocals_mel_band_roformer.ckpt"
# None = audio-separator's own model cache
DEFAULT_STEM_MODEL_DIR = os.environ.get("UTASUB_STEM_MODEL_DIR") or None


def asr_available():
  """True when the ASR model package is importable."""
  try:
    import qwen_asr  # noqa: F401
    return True
  except ImportError:
    return False


# --- silence chunking (clean vocals dip; full mix falls back to one chunk) ---

def split_on_silence(audio, sr=16000, min_gap=0.5, min_voice=0.3, pad=0.15, max_chunk=30.0):
  import numpy as np
  frame = int(sr * 0.05)
  n = len(audio) // frame
  dur = len(audio) / sr
  if n == 0:
    return [(0.0, dur)]
  rms = np.sqrt((audio[:n * frame].reshape(n, frame).astype(np.float64) ** 2).mean(axis=1))
  voiced = rms > max(float(rms.max()) * 0.02, 1e-3)

  regions = []
  start = None
  last = None
  for i, v in enumerate(voiced):
    if not v:
      continue
    if start is None:
      start = i
    elif (i - last) * 0.05 >= min_gap:
      regions.append((start, last + 1))
      start = i
    last = i
  if start is not None:
    regions.append((start, last + 1))

  chunks = []
  for s, e in regions:
    t0 = max(0.0, s * 0.05 - pad)
    t1 = min(dur, e * 0.05 + pad)
    if t1 - t0 < min_voice:
      continue
    while t1 - t0 > max_chunk:  # hard split; upgrade: cut at quietest frame
      chunks.append((t0, t0 + max_chunk))
      t0 += max_chunk
    chunks.append((t0, t1))
  return chunks or [(0.0, dur)]


# --- segment grouping ---

SENTENCE_ENDS = set("。！？.!?")


def smart_join(tokens):
  # space between two ASCII-word boundaries only: EN keeps spacing, CJK stays fused
  out = ""
  for t in tokens:
    if out and t and (out[-1].isascii() and out[-1].isalnum()) and (t[0].isascii() and t[0].isalnum()):
      out += " "
    out += t
  return re.sub(r" +", " ", out)


def group_into_segments(stamps, max_duration=8.0, max_chars=40, max_gap=0.8):
  segments = []
  buf = []
  buf_start = None
  buf_end = 0.0

  def flush():
    nonlocal buf, buf_start
    text = smart_join(buf).strip()
    if text and buf_start is not None:
      segments.append((buf_start, buf_end, text))
    buf = []
    buf_start = None

  for w in stamps:
    if buf and (w.start_time - buf_end) > max_gap:  # silence gap = phrase boundary
      flush()
    if buf_start is None:
      buf_start = w.start_time
    buf.append(w.text)
    buf_end = w.end_time
    text = smart_join(buf)

    at_break = text and text[-1] in SENTENCE_ENDS
    too_long = (buf_end - buf_start) > max_duration
    too_wide = len(text) > max_chars
    if at_break or too_long or too_wide:
      flush()

  flush()
  return segments


# --- stem separation (vocals only, FLAC, cached) ---

def find_instrumental(media_path):
  """Cached instrumental stem <name>.instrumental.*, or None.
  Optional: everything works with vocals stem alone."""
  want = f"{Path(media_path).stem}.instrumental".lower()
  cache = stem_cache_dir()
  return next((f for f in cache.iterdir()
               if f.is_file() and f.stem.lower() == want), None)


def extract_vocals(media_path, stem_model, model_dir, progress):
  """Separate stems into stem cache as <name>.vocals.flac and
  <name>.instrumental.flac. FLAC keeps them small; find_vocals picks up vocals
  for FA later, instrumental sits there for future energy-ratio signals
  (MC detection), never required.
  Returns cached vocals path, or None when audio-separator unavailable."""
  try:
    from audio_separator.separator import Separator
  except ImportError:
    progress("  audio-separator not installed, transcribing full mix")
    return None
  out_dir = stem_cache_dir()
  # both stems from 2-stem model; model_file_dir omitted when unset, so
  # audio-separator uses its own cache
  sep = Separator(output_dir=str(out_dir), output_format="FLAC",
                  **({"model_file_dir": model_dir} if model_dir else {}))
  sep.load_model(stem_model)
  outputs = sep.separate(str(media_path))
  stem = Path(media_path).stem
  vocals = None
  for name in outputs:
    produced = Path(out_dir) / name
    # default to vocals: single-stem model names its one output freely
    kind = "instrumental" if "instrument" in name.lower() else "vocals"
    dest = out_dir / f"{stem}.{kind}.flac"
    if produced.resolve() != dest.resolve():
      produced.replace(dest)
    if kind == "vocals":
      vocals = dest
  return vocals


# --- model ---

def _load_model(small, cpu, progress):
  import torch
  from transformers.utils import logging as hf_logging
  hf_logging.set_verbosity_error()  # silence temperature/pad_token_id chatter
  from qwen_asr import Qwen3ASRModel
  model_id = "Qwen/Qwen3-ASR-0.6B" if small else "Qwen/Qwen3-ASR-1.7B"
  dtype = torch.float32 if cpu else torch.bfloat16
  device = "cpu" if cpu else "cuda:0"
  progress(f"  loading {model_id}...")
  return Qwen3ASRModel.from_pretrained(
    model_id, dtype=dtype, device_map=device,
    # small batches + tight token cap: a music chunk can repetition-loop to
    # token limit and peg GPU/VRAM if all chunks batch at once
    max_inference_batch_size=4, max_new_tokens=512,
    # required by return_time_stamps=True: Qwen stamps words via the aligner
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(dtype=dtype, device_map=device))


# --- top-level ---

def transcribe(media_path, *, lang=None, small=False, cpu=False, use_stems=True,
               stem_model=DEFAULT_STEM_MODEL, stem_model_dir=DEFAULT_STEM_MODEL_DIR,
               overwrite=False, progress=print):
  """Transcribe media, write ASR block into <name>.utasub.json. Returns session
  path, or None on skip.
  Reuses existing vocal stem; separates one when use_stems and none present."""
  media_path = Path(media_path)
  if session.has_asr(media_path) and not overwrite:
    progress(f"  skip (asr exists): {media_path.name}")
    return session.session_path(media_path)

  start = time.time()
  progress(f"=== {media_path.name}")

  audio_path = media_path
  if use_stems:
    existing = find_vocals(media_path)
    if existing:
      audio_path = existing
      progress(f"  vocal stem (reused): {existing.name}")
    else:
      progress(f"  separating vocals ({stem_model})...")
      vocal = extract_vocals(media_path, stem_model, stem_model_dir, progress)
      if vocal:
        audio_path = vocal
        progress(f"  vocal stem: {vocal.name}")

  lang_name = LANG_CODES_TO_NAMES.get(lang) if lang else None
  progress("  decoding audio...")
  audio = load_audio_16k(audio_path)
  chunks = split_on_silence(audio)
  progress(f"  audio: {len(audio) / 16000:.0f}s, {len(chunks)} chunk(s)")

  progress("  transcribing...")
  model = _load_model(small, cpu, progress)
  results = model.transcribe(
    audio=[(audio[int(cs * 16000):int(ce * 16000)], 16000) for cs, ce in chunks],
    language=lang_name, return_time_stamps=True)

  segments = []
  langs = []
  for (chunk_start, chunk_end), result in zip(chunks, results):
    if result.language:
      langs.append(result.language)
    cjk = is_cjk(result.text)  # tighter line width for CJK
    subs = group_into_segments(result.time_stamps, max_chars=40 if cjk else 84) if result.time_stamps else []
    # aligner drifts on singing: zero-length/fragment stamps trust chunk bounds instead
    degenerate = not subs or any(e - s < 0.2 for s, e, _ in subs)
    if degenerate:
      if result.text.strip():
        segments.append((chunk_start, chunk_end, result.text.strip()))
    else:
      for s, e, text in subs:
        segments.append((s + chunk_start, e + chunk_start, text))
  detected_lang = max(set(langs), key=langs.count) if langs else ""

  out_path = session.save_asr(media_path, detected_lang, segments)

  full_text = "".join(t for _, _, t in segments)
  progress(f"  language: {detected_lang}, {len(full_text)} chars, {len(segments)} segments")
  progress(f"  -> {out_path.name}  ({time.time() - start:.0f}s)")
  return out_path


# --- CLI (utasub-asr) ---

def main():
  import argparse
  import sys
  if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  parser = argparse.ArgumentParser(description="Transcribe media (ASR block in .utasub.json) via Qwen3-ASR")
  parser.add_argument("targets", nargs="+", help="media files")
  parser.add_argument("--lang", default=None, help="force language code (default: auto)")
  parser.add_argument("--small", action="store_true", help="0.6B model (faster, less accurate)")
  parser.add_argument("--cpu", action="store_true", help="force CPU inference")
  parser.add_argument("--no-stems", dest="stems", action="store_false",
                      help="transcribe full mix instead of separating vocals")
  parser.add_argument("--overwrite", action="store_true", help="redo files that already have an ASR block")
  args = parser.parse_args()
  for target in args.targets:
    transcribe(target, lang=args.lang, small=args.small, cpu=args.cpu,
               use_stems=args.stems, overwrite=args.overwrite)


if __name__ == "__main__":
  main()
