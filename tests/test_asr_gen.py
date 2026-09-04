"""Tests for ASR generation: batched transcription progress and chunk ordering."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from utasub.core import asr_gen

class _FakeModel:
  """Returns one result per input chunk; text encodes call order so tests can
  check segments come out in chunk order. Empty time_stamps forces the
  chunk-bounds (degenerate) path, one segment per chunk."""

  def __init__(self):
    self.seen = 0

  def transcribe(self, audio, language=None, return_time_stamps=False):
    results = []
    for _ in audio:
      results.append(SimpleNamespace(language="ja", text=f"c{self.seen}", time_stamps=[]))
      self.seen += 1
    return results

def _transcribe(n_chunks):
  chunks = [(float(i), float(i + 1)) for i in range(n_chunks)]
  audio = np.zeros(n_chunks * 16000, dtype=np.float32)
  lines = []
  captured = {}

  def save_asr(media_path, lang, segments):
    captured["lang"] = lang
    captured["segments"] = segments
    from pathlib import Path
    return Path("out.utasub.json")

  with patch.object(asr_gen, "_load_model", return_value=_FakeModel()), \
       patch.object(asr_gen, "load_audio_16k", return_value=audio), \
       patch.object(asr_gen, "split_on_silence", return_value=chunks), \
       patch.object(asr_gen.session, "has_asr", return_value=False), \
       patch.object(asr_gen.session, "save_asr", side_effect=save_asr):
    asr_gen.transcribe("song.mkv", use_stems=False, progress=lines.append)
  return lines, captured

def test_progress_one_line_per_batch():
  """45 chunks at batch size 20 emit 3 progress lines with running counts."""
  lines, _ = _transcribe(45)
  chunk_lines = [l for l in lines if "chunks," in l and "left" in l]
  assert len(chunk_lines) == 3, chunk_lines
  counts = [l.strip().split("/")[0] for l in chunk_lines]
  assert counts == ["20", "40", "45"]
  assert all("/45 chunks" in l for l in chunk_lines)

def test_single_batch_when_under_batch_size():
  """Fewer chunks than batch size emit exactly one progress line."""
  lines, _ = _transcribe(5)
  chunk_lines = [l for l in lines if "chunks," in l and "left" in l]
  assert len(chunk_lines) == 1
  assert "5/5 chunks" in chunk_lines[0]

def test_segments_come_out_in_chunk_order():
  """Batched results reassemble in chunk order across batch boundaries."""
  _, captured = _transcribe(45)
  segments = captured["segments"]
  assert len(segments) == 45
  assert [t for _, _, t in segments] == [f"c{i}" for i in range(45)]
  starts = [s for s, _, _ in segments]
  assert starts == sorted(starts)
  assert captured["lang"] == "ja"
