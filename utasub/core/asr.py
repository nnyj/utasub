"""Read the cached ASR block from the merged <name>.utasub.json session file.
Tool never runs ASR; asr_gen writes the block, this reads it."""
from .session import _read_raw

def load_asr(media_path):
  """ASR block from the session file next to media_path.
  Returns (segments, language) where segments = [(start, end, text), ...],
  or None when no ASR block is present."""
  data = _read_raw(media_path)
  asr = data.get("asr") if data else None
  if not asr:
    return None
  segments = [(s, e, t) for s, e, t in asr["segments"]]
  return segments, asr.get("language", "")
