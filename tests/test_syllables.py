"""Unit tests for the syllabize core helper: token_spans relative to cue start,
bounds untouched, and the no-stamp / short-window failure paths. Aligner stubbed,
no GPU/torchaudio needed."""
import math

import numpy as np

from utasub.core.align import Cue
from utasub.core.syllables import syllabize_cue

SR = 100  # tiny rate keeps the sliced windows small

def _audio(dur_s):
  return np.arange(int(dur_s * SR), dtype=np.float32)

def _stub(captured, score=-0.1, spans=(("a", 0.5, 0.7),)):
  """Records the window length it saw, then reports a fixed stamp."""
  def _a(text, win):
    captured["len"] = len(win)
    return (0.5, 1.5, score, list(spans))
  return _a

def test_window_is_the_cue_bounds():
  cue = Cue(10.0, 12.0, "sample lyric")
  cap = {}
  res = syllabize_cue(cue, _audio(30.0), SR, aligner=_stub(cap))
  assert cap["len"] == int(12.0 * SR) - int(10.0 * SR)  # exactly [start, end]
  assert res.token_spans == [("a", 0.5, 0.7)]  # already relative to cue.start
  assert res.score == -0.1

def test_low_score_still_adopts():
  """Recalc is explicit + undoable: adopt the best stamp even at a low score."""
  cue = Cue(10.0, 12.0, "x")
  low = math.log(0.02)
  res = syllabize_cue(cue, _audio(30.0), SR, aligner=_stub({}, score=low))
  assert res is not None and res.score == low

def test_none_stamp_fails():
  cue = Cue(10.0, 12.0, "x")
  assert syllabize_cue(cue, _audio(30.0), SR, aligner=lambda t, w: None) is None

def test_short_window_fails():
  cue = Cue(10.0, 10.1, "x")  # 0.1s span, under MIN_WIN
  assert syllabize_cue(cue, _audio(30.0), SR, aligner=_stub({})) is None
