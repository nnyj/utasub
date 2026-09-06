"""Tests for ASR generation: batched transcription progress and chunk ordering."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

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

@pytest.mark.parametrize("n_chunks, n_lines, counts", [
  (45, 3, ["20", "40", "45"]),   # batch size 20: two full batches plus remainder
  (5, 1, ["5"]),                 # under batch size: single batch
])
def test_progress_one_line_per_batch(n_chunks, n_lines, counts):
  """One progress line per batch, carrying running done/total counts."""
  lines, _ = _transcribe(n_chunks)
  chunk_lines = [l for l in lines if "chunks," in l and "left" in l]
  assert len(chunk_lines) == n_lines, chunk_lines
  assert [l.strip().split("/")[0] for l in chunk_lines] == counts
  assert all(f"/{n_chunks} chunks" in l for l in chunk_lines)

def test_segments_come_out_in_chunk_order():
  """Batched results reassemble in chunk order across batch boundaries."""
  _, captured = _transcribe(45)
  segments = captured["segments"]
  assert len(segments) == 45
  assert [t for _, _, t in segments] == [f"c{i}" for i in range(45)]
  starts = [s for s, _, _ in segments]
  assert starts == sorted(starts)
  assert captured["lang"] == "ja"

def _tone(seconds, sr=16000):
  t = np.arange(int(seconds * sr)) / sr
  return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

def test_split_on_silence_cuts_at_the_gap():
  """1s tone, 1s silence, 1s tone: two chunks, boundaries padded by 0.15s."""
  audio = np.concatenate([_tone(1), np.zeros(16000, dtype=np.float32), _tone(1)])
  chunks = asr_gen.split_on_silence(audio)
  assert len(chunks) == 2, chunks
  assert chunks[0] == pytest.approx((0.0, 1.15), abs=0.01)
  assert chunks[1] == pytest.approx((1.85, 3.0), abs=0.01)

def test_split_on_silence_returns_whole_span_when_nothing_voiced():
  chunks = asr_gen.split_on_silence(np.zeros(32000, dtype=np.float32))
  assert chunks == [(0.0, 2.0)]

def test_group_into_segments_splits_on_word_gap():
  """Gap wider than max_gap ends the phrase; words join with ASCII spacing."""
  stamps = [SimpleNamespace(start_time=s, end_time=e, text=t) for s, e, t in [
    (0.0, 0.5, "Hello"), (0.5, 1.0, "world"),
    (2.5, 3.0, "second"), (3.0, 3.5, "phrase"),
  ]]
  assert asr_gen.group_into_segments(stamps) == [
    (0.0, 1.0, "Hello world"), (2.5, 3.5, "second phrase")]
