"""Fill one cue's karaoke syllable timing by CTC-aligning its text inside its
current bounds. Bounds are never moved: the user owns start/end, this only
computes token_spans (relative to cue start) and the confidence score within
[start, end]. Drives the grid's Recalc syllables action for cues the placement
chain left without a stamp (blank Conf, no karaoke fill)."""
from collections import namedtuple

MIN_WIN = 0.3  # a window shorter than this can't be aligned (align_lines floor)

SyllableResult = namedtuple("SyllableResult", "score token_spans")

def ctc_line(text, audio_win):
  """Real single-line aligner over 16k audio: (start, end, score,
  [(char, start, end)]) in window-local seconds, or None when nothing stamps."""
  from .ctc_align import align_lines
  starts, ends, scores, spans = align_lines([text], audio_win)
  if 0 not in starts:
    return None
  return starts[0], ends[0], scores.get(0), spans.get(0, [])

def syllabize_cue(cue, audio, sr, aligner=ctc_line):
  """CTC-align cue text inside [cue.start, cue.end] and return a SyllableResult
  (score + token_spans relative to cue.start), or None when the window is too
  short or the aligner stamps nothing. Bounds stay put: the window is the cue's
  own span, so spans are already relative to cue.start. Pure: audio is sliced
  here, `aligner(text, window)` does the CTC work, so a stub drives the test."""
  lo = cue.start
  hi = min(cue.end, len(audio) / sr)
  if hi - lo < MIN_WIN:
    return None
  res = aligner(cue.text, audio[int(lo * sr):int(hi * sr)])
  if res is None or res[2] is None:
    return None
  _start, _end, score, spans = res
  rebased = [(c, round(a, 3), round(b, 3)) for c, a, b in spans]
  return SyllableResult(score, rebased or None)
