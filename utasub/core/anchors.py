"""Anchor collection for the LRC warp: one global monotonic CTC pass.

`ctc_align` runs a single Viterbi path over the whole region span, so every
line is stamped in order and on its own occurrence of a repeated lyric.
No seed, no windowing needed.

Anchors handed over ungated: `fit_warp`'s monotone/linear chain filters
already discard outliers, so gating here would only thin the fit. Confidence
gate lives where stamps are consumed directly (`place._adopt_fa`).
"""

def collect(lrc_times, texts, audio):
  """(candidate anchor sets, stamp bundle) in window-local audio time.
  Candidates are [(anchors, diag)] with anchors = [(line_idx, audio_time)].
  The bundle is (starts, ends, scores) from the same pass, so the placement's
  polish reuses it without aligning again. Both empty when nothing stamped."""
  from .ctc_align import align_lines
  starts, ends, scores = align_lines(texts, audio)
  anchors = sorted(starts.items())
  if len(anchors) < 3:
    return [], (starts, ends, scores)
  diag = {"aligner": "ctc", "stamps": len(anchors)}
  return [(anchors, diag)], (starts, ends, scores)
