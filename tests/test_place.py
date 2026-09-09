"""Unit tests for placement-strategy chain: offset estimation, checkpoint majority,
chain fallthrough. No GPU needed (monkeypatched aligners)."""
import pytest

from utasub.core.ctc_align import low_conf_cut

def _make_lines(n=20, spacing=5.0, offset=0.0):
  """Synthetic timed LRC lines."""
  return [(offset + i * spacing, f"line {i} text here") for i in range(n)]

def _make_segments(lines, jitter=0.2):
  """Fake ASR segments from LRC lines with small jitter."""
  import random
  random.seed(99)
  segs = []
  for t, txt in lines:
    s = max(0.0, t + random.uniform(-jitter, jitter))
    e = s + 3.0
    segs.append((s, e, txt.replace("line", "lyne")))  # garbled
  return segs

def _fake_audio(duration_s=120.0, sr=16000):
  """Silent audio array."""
  import numpy as np
  return np.zeros(int(duration_s * sr), dtype=np.float32)

# --- monkeypatch the aligners ---

_fa_offset = 1.5  # the aligners return LRC times + this offset

def _fa_truth(truth):
  """Qwen FA mock (coarse_fa path) reporting true audio time regardless of seed."""
  def _align(starts, texts, audio, sr=16000, pre_pad=5.0, post_pad=10.0,
             drift_gate=8.0):
    return ({j: t for j, t in enumerate(truth)},
            {j: t + 2.0 for j, t in enumerate(truth)})
  return _align

def _patch_ctc(monkeypatch, truth, score=-0.5):
  """Stand in for the global CTC pass: stamps every line at its true time."""
  from utasub.core import ctc_align

  def _align_lines(texts, audio):
    return ({j: t for j, t in enumerate(truth)},
            {j: t + 2.0 for j, t in enumerate(truth)},
            {j: score for j, _ in enumerate(truth)}, {})
  monkeypatch.setattr(ctc_align, "align_lines", _align_lines)
  monkeypatch.setattr(ctc_align, "ctc_available", lambda: True)

def _mock_force_align_lines(starts, texts, audio, sr=16000, pre_pad=5.0,
                            post_pad=10.0, drift_gate=8.0):
  """Seed-relative mock, for the coarse paths that only run FA once."""
  fa_starts = {j: starts[j] + _fa_offset for j in range(len(starts))}
  fa_ends = {j: starts[j] + _fa_offset + 2.0 for j in range(len(starts))}
  return fa_starts, fa_ends

def _mock_fa_available():
  return True

def _patch_fa(monkeypatch, align_fn, available_fn=_mock_fa_available):
  """Swap fa.force_align_lines / fa.fa_available; auto-restored by monkeypatch."""
  from utasub.core import fa
  monkeypatch.setattr(fa, "force_align_lines", align_fn)
  monkeypatch.setattr(fa, "fa_available", available_fn)

# --- warp fit ---

def test_steady_offset_fits_one_segment(monkeypatch):
  """Steady offset: one segment, LRC intervals reproduced exactly."""
  lines = _make_lines(20, spacing=5.0)
  _patch_ctc(monkeypatch, [t + _fa_offset for t, _ in lines])
  from utasub.core.place import _lrc_warp
  result = _lrc_warp(lines, _make_segments(lines), _fake_audio(200.0), use_fa=True)
  assert result is not None, "_lrc_warp returned None on a clean offset"
  cues, _, meta = result
  assert meta["strategy"] == "lrc_warp"
  assert meta["segments"] == 1, f"expected 1 segment, got {meta['segments']}"
  starts = [c.start for c in cues if hasattr(c, "confidence") and c.confidence]
  assert abs(starts[0] - (lines[0][0] + _fa_offset)) < 0.3
  assert abs((starts[5] - starts[4]) - 5.0) < 0.1, "LRC interval not preserved"

def test_warp_fits_inserted_gap():
  """An inserted instrumental gets a second segment, but only where the audio is silent."""
  from utasub.core.warp import fit_warp
  lrc = [i * 5.0 for i in range(20)]
  # lines 0-9 at +2s, lines 10-19 at +32s (30s of extra instrumental)
  anchors = [(j, lrc[j] + (2.0 if j < 10 else 32.0)) for j in range(20)]
  runs = [(2.0, 47.0), (82.0, 130.0)]  # 35s of silence covers the insert
  fitted = fit_warp(anchors, lrc, runs)
  assert fitted is not None
  segs, diag = fitted
  assert diag["segments"] == 2, f"expected 2 segments, got {diag['segments']}"
  assert abs(diag["jumps"][0] - 30.0) < 1.0, diag["jumps"]

  # same anchors, continuously voiced audio: the jump has nowhere to hide
  segs2, diag2 = fit_warp(anchors, lrc, [(2.0, 130.0)])
  assert diag2["segments"] == 1, "jump accepted without a silent seam"

def test_warp_recovers_steep_slope():
  """A take 24% slower than the studio cut fits as one steep segment, not a stack
  of fake gap insertions."""
  from utasub.core.warp import apply_warp, fit_warp
  lrc = [i * 4.5 for i in range(40)]
  anchors = [(j, 1.238 * lrc[j] + 20.0) for j in range(40)]
  fitted = fit_warp(anchors, lrc, [(20.0, 240.0)])
  assert fitted is not None
  segs, diag = fitted
  assert diag["segments"] == 1, f"expected 1 segment, got {diag['segments']}: {diag}"
  assert abs(diag["slopes"][0] - 1.238) < 0.02, diag["slopes"]
  assert not diag["jumps"], diag["jumps"]
  placed = apply_warp(lrc, segs)
  assert max(abs(p - a) for p, (_, a) in zip(placed, anchors)) < 0.5

def test_warp_rejects_break_into_unmeasured_slope():
  """A seam into a tail too sparse to measure a slope is refused."""
  from utasub.core.warp import fit_warp
  # dense head measures 1.238; the 5-anchor tail sits on a slope-1 line, too
  # short to measure, and fitting it flat beats extending the head
  head = [i * 4.5 for i in range(30)]
  tail = [140.0, 150.0, 160.0, 170.0, 180.0]
  lrc = head + tail
  anchors = [(j, 1.238 * head[j] + 20.0) for j in range(30)]
  seam = 1.238 * 140.0 + 20.0
  anchors += [(30 + k, seam + (t - 140.0)) for k, t in enumerate(tail)]
  segs, diag = fit_warp(anchors, lrc, [(20.0, 280.0)])
  assert diag["segments"] == 1, (
    f"broke into an unmeasured segment: {diag['segments']} segs, {diag['slopes']}")
  assert abs(diag["slopes"][0] - 1.238) < 0.02, diag["slopes"]

def test_warp_rejects_a_tempo_below_the_slope_floor():
  """Tail slope under SLOPE_LO is a fitter-invented tempo drop; rejecting it
  leaves tail unmeasured, seam gate collapses to one segment."""
  from utasub.core.warp import SLOPE_LO, fit_warp
  lrc = [i * 4.5 for i in range(30)] + [140.0 + 8.0 * i for i in range(8)]
  # whole song runs at 1.036; the tail anchors are early enough that an
  # unconstrained fit would call it a 0.85 slowdown, below any real take
  anchors = [(j, 1.036 * lrc[j] + 30.0) for j in range(30)]
  anchors += [(30 + k, 0.85 * lrc[30 + k] + 56.0) for k in range(8)]
  segs, diag = fit_warp(anchors, lrc, [(30.0, 290.0)])
  assert diag["segments"] == 1, f"invented a tempo drop: {diag['slopes']}"
  assert all(s >= SLOPE_LO for s in diag["slopes"]), diag["slopes"]

def test_collect_returns_the_global_pass_ungated(monkeypatch):
  """collect hands warp every stamp, gates nothing: fit_warp's own chain
  filters do that job."""
  lines = _make_lines(20, spacing=5.0)
  truth = [t + _fa_offset for t, _ in lines]
  _patch_ctc(monkeypatch, truth)
  from utasub.core import ctc_align
  from utasub.core.place import AlignOpts, run_chain

  # one bad-confidence line: still an anchor, and still gated out of direct use
  def _align_lines(texts, audio):
    return ({j: t for j, t in enumerate(truth)},
            {j: t + 2.0 for j, t in enumerate(truth)},
            {j: (-9.0 if j == 4 else -0.5) for j in range(len(truth))}, {})
  monkeypatch.setattr(ctc_align, "align_lines", _align_lines)
  _patch_fa(monkeypatch, _fa_truth(truth))
  cues, _, meta = run_chain(lines, _make_segments(lines), _fake_audio(200.0),
                            AlignOpts(use_fa=True))
  assert meta["strategy"] == "lrc_warp", meta
  assert meta["stamps"] == 20 and meta["aligner"] == "ctc"
  assert meta["gated"] == 19, "gate kept the low-confidence line"

  # too few stamps to fit: warp declines
  monkeypatch.setattr(ctc_align, "align_lines",
                      lambda texts, audio: ({0: 1.0}, {0: 3.0}, {0: -0.5}, {}))
  _, _, meta = run_chain(lines, _make_segments(lines), _fake_audio(200.0),
                         AlignOpts(use_fa=True))
  assert meta["strategy"] != "lrc_warp", meta

def test_adopt_fa_takes_gated_stamps_within_the_bound():
  """Every gated stamp near placement is taken outright; one a second away is
  a smeared line, left alone. Order survives either way."""
  from utasub.core.place import ADOPT_MAX, _adopt_fa
  base = [10.0, 20.0, 30.0]
  out = _adopt_fa(base, {0: 9.0, 2: 30.0 - ADOPT_MAX - 1.0})
  assert out[0] == 9.0
  assert out[1] == 20.0, "an ungated line was moved"
  assert out[2] == 30.0, "adopted a stamp past the bound"
  # a within-bound stamp that would cross its predecessor is repaired, not refused
  assert _adopt_fa([10.0, 11.0], {1: 9.5})[1] == pytest.approx(10.05)
  assert _adopt_fa(base, {}) == base

def test_warp_preserves_monotonicity():
  """apply_warp never goes backwards, whatever the segments say."""
  from utasub.core.warp import apply_warp, warped_durations
  lrc = [0.0, 5.0, 10.0, 15.0]
  segs = [(0, 1.0, 2.0), (2, 1.0, 12.0)]
  out = apply_warp(lrc, segs)
  assert out == [2.0, 7.0, 22.0, 27.0]
  assert all(b >= a for a, b in zip(out, out[1:]))

  # a stretched segment scales both the placement and the cue durations
  segs2 = [(0, 1.0, 2.0), (2, 1.2, 12.0)]
  assert apply_warp(lrc, segs2) == [2.0, 7.0, 24.0, 30.0]
  assert warped_durations(lrc, segs2)[:3] == [5.0, 5.0, 6.0]

def _no_ctc_stamps(monkeypatch):
  _patch_ctc(monkeypatch, [])

def _ctc_unavailable(monkeypatch):
  """torchaudio absent: no anchors possible."""
  from utasub.core import ctc_align
  monkeypatch.setattr(ctc_align, "ctc_available", lambda: False)

LRC_WARP_REJECT_CASES = [
  # no anchors: no CTC stamps, nothing to fit, with and without an envelope
  (_no_ctc_stamps, True, {"use_fa": True}),
  (_no_ctc_stamps, True, {"use_fa": True, "runs_lo": [(3.0, 60.0)]}),
  # aligner missing
  (_ctc_unavailable, True, {"use_fa": True, "runs_lo": [(3.0, 60.0)]}),
  # FA off: nothing can fit a warp
  (None, True, {"use_fa": False, "runs_lo": [(3.0, 60.0)]}),
  # no audio at all
  (None, False, {"use_fa": True}),
]

@pytest.mark.parametrize("setup,audio,kwargs", LRC_WARP_REJECT_CASES,
                         ids=["no_anchors", "no_anchors_runs", "no_ctc",
                              "no_fa", "no_audio"])
def test_lrc_warp_rejects_and_the_chain_falls_through(monkeypatch, setup, audio,
                                                      kwargs):
  if setup:
    setup(monkeypatch)
  from utasub.core.place import _lrc_warp
  lines = _make_lines(12, spacing=5.0)
  assert _lrc_warp(lines, _make_segments(lines),
                   _fake_audio(120.0) if audio else None, **kwargs) is None

def test_lrc_warp_needs_no_transcript(monkeypatch):
  """Global pass takes no seed: region with no usable transcript still places."""
  lines = _make_lines(12, spacing=5.0, offset=1.0)
  _patch_ctc(monkeypatch, [t + _fa_offset for t, _ in lines])
  from utasub.core.place import _lrc_warp
  result = _lrc_warp(lines, [], _fake_audio(120.0), use_fa=True,
                     runs_lo=[(4.0, 60.0)])
  assert result is not None, "no-transcript region rejected"
  assert result[2]["strategy"] == "lrc_warp"

def test_chain_fallthrough(monkeypatch):
  """lrc_warp wins when it fits; chain falls to coarse_fa when it can't;
  coarse_fa drops to envelope when FA stamps too little."""
  lines = _make_lines(20, spacing=5.0)
  _patch_ctc(monkeypatch, [t + _fa_offset for t, _ in lines])
  _patch_fa(monkeypatch, _fa_truth([t + _fa_offset for t, _ in lines]))
  from utasub.core import place
  from utasub.core.place import run_chain, AlignOpts, _coarse_fa
  segments = _make_segments(lines)
  audio = _fake_audio(200.0)

  _, _, meta = run_chain(lines, segments, audio, AlignOpts(use_fa=True))
  assert meta["strategy"] == "lrc_warp", f"expected lrc_warp, got {meta['strategy']}"

  # take lrc_warp out of the chain: the next strategy must carry the placement
  monkeypatch.setattr(place, "CHAIN",
                      [row for row in place.CHAIN if row[0] != "lrc_warp"])
  _, _, meta2 = run_chain(lines, segments, audio, AlignOpts(use_fa=True))
  assert meta2["strategy"] == "coarse_fa", f"expected coarse_fa, got {meta2['strategy']}"

  # envelope only, no FA
  _, _, meta3 = _coarse_fa(lines, segments, audio, use_fa=False)
  assert meta3["strategy"] == "coarse_envelope"

  # FA that stamps almost nothing degrades to that same path from inside coarse_fa
  _patch_fa(monkeypatch, _fa_truth([lines[0][0] + _fa_offset]))
  _, _, meta4 = _coarse_fa(lines, segments, audio, use_fa=True)
  assert meta4["strategy"] == "coarse_envelope", \
    f"sparse FA should fall to the envelope, got {meta4['strategy']}"

  # untimed lines cannot warp, so the chain skips lrc_warp
  _patch_fa(monkeypatch, _fa_truth([t + _fa_offset for t, _ in lines]))
  untimed = [(None, t) for _, t in lines]
  _, _, meta5 = run_chain(untimed, segments, audio, AlignOpts(use_fa=True))
  assert meta5["strategy"] in ("coarse_fa", "coarse_envelope"), \
    f"untimed should skip lrc_warp, got {meta5['strategy']}"

def test_rescue_from_silence_moves_forward_only():
  """A start stranded in an unvoiced gap is carried forward to next voiced run:
  never backward, never out of voice, never past SILENCE_REACH."""
  from utasub.core.place import SILENCE_REACH, _rescue_from_silence
  runs = [(10.0, 14.0), (20.0, 24.0), (40.0, 44.0)]

  # stranded past a singer's natural lead-in: carried forward onto the onset
  assert _rescue_from_silence(18.2, runs) == 20.0
  # inside the plausible lead-in: left alone
  assert _rescue_from_silence(18.8, runs) == 18.8
  # already inside a voiced run: untouched
  assert _rescue_from_silence(12.0, runs) == 12.0
  # late, sitting in the gap after a run: forward only, so it stays put
  assert _rescue_from_silence(30.0, runs) == 30.0
  # next voice further than reach: left where it is
  assert _rescue_from_silence(20.0 + SILENCE_REACH + 5.0, runs) == 20.0 + SILENCE_REACH + 5.0
  # no envelope at all: nothing to rescue against
  assert _rescue_from_silence(18.2, []) == 18.2

def test_linear_anchors_drops_wrong_occurrence_stamps():
  """A stamp matched to an earlier repeat lands in order; only the impossible
  tempo it implies on both sides separates it."""
  from utasub.core.warp import linear_anchors
  lrc = [i * 5.0 for i in range(20)]
  anchors = [(j, 20.0 + 1.05 * lrc[j]) for j in range(20)]
  # lines 7 and 13 landed 4s early on a repeat, still in ascending audio order
  anchors[7] = (7, anchors[7][1] - 4.0)
  anchors[13] = (13, anchors[13][1] - 4.0)
  assert all(a[1] <= b[1] for a, b in zip(anchors, anchors[1:]))
  kept = {j for j, _ in linear_anchors(sorted(anchors), lrc)}
  assert 7 not in kept and 13 not in kept, f"kept a wrong-occurrence stamp: {kept}"
  assert len(kept) == 18, f"dropped a good anchor too: {sorted(kept)}"

def test_linear_anchors_keeps_a_take_that_inserted_material():
  """Anchors costing most of the set are a real insertion, not noise: leave
  them for the changepoint fit."""
  from utasub.core.warp import linear_anchors
  lrc = [i * 5.0 for i in range(20)]
  # 30s of extra instrumental halfway through: two clean halves, one real seam
  anchors = [(j, 20.0 + lrc[j] + (0.0 if j < 10 else 30.0)) for j in range(20)]
  kept = linear_anchors(anchors, lrc)
  assert len(kept) == 20, f"discarded a real gap insertion, kept {len(kept)}"

@pytest.mark.parametrize("extra_line", [None, 299.0])
def test_cut_preserves_order_between_unanchored_lines(extra_line):
  from utasub.core.warp import _segment_fit, apply_warp
  times = [float(t) for t in range(0, 210, 10)] + [float(t) for t in range(300, 600, 10)]
  if extra_line is not None:
    times = sorted(times + [extra_line])
  anchors = [(j, t if t <= 200 else t - 30) for j, t in enumerate(times)
             if t <= 200 or t >= 300]
  fitted = _segment_fit(anchors, times, [(200.0, 270.0)])
  segments = [(anchors[a][0], slope, offset) for a, _, slope, offset in fitted]
  starts = apply_warp(times, segments)
  assert all(a <= b for a, b in zip(starts, starts[1:]))
  if extra_line is None:
    assert len(segments) == 2
    assert starts[-1] == pytest.approx(times[-1] - 30)

def test_partial_timestamps_use_coarse_placement(monkeypatch):
  from utasub.core.place import AlignOpts, run_chain
  from utasub.core.align import Cue
  lines = _make_lines(12)
  segments = _make_segments(lines)
  truth = [t + _fa_offset for t, _ in lines]
  _patch_ctc(monkeypatch, truth)
  _patch_fa(monkeypatch, _fa_truth(truth))
  lines[3] = (None, lines[3][1])
  cues, span, meta = run_chain(lines, segments, _fake_audio(), AlignOpts())
  assert meta["strategy"] == "coarse_fa"
  assert span is not None
  assert len([c for c in cues if isinstance(c, Cue)]) == len(lines)

def test_ctc_polish_keeps_short_cues_nonoverlapping(monkeypatch):
  from utasub.core.align import Cue
  from utasub.core.place import _ctc_polish
  _patch_ctc(monkeypatch, [1.0, 1.05])
  cues = [Cue(1.0, 2.0, "first"), Cue(1.05, 2.0, "second")]
  _ctc_polish(cues, _fake_audio())
  assert cues[0].start < cues[0].end <= cues[1].start
  assert cues[1].start < cues[1].end

# --- gap 5 / gap 3: per-cue CTC evidence ---

def test_low_conf_cut_is_a_per_song_percentile():
  """Absolute scores move with the mix and the singer, the ranking inside one
  song does not, so the cut is drawn from the song's own scores."""
  scores = [0.1 * i for i in range(1, 21)]
  cut = low_conf_cut(scores, pct=10)
  assert sum(1 for s in scores if s < cut) == 2, cut
  quiet = [s * 0.1 for s in scores]  # same song mixed 10x quieter
  assert sum(1 for s in quiet if s < low_conf_cut(quiet, pct=10)) == 2

def test_low_conf_cut_ignores_unstamped_cues():
  assert low_conf_cut([0.9, None, 0.8, None, 0.7, 0.6, 0.5]) == 0.5

def test_low_conf_cut_is_relative_only():
  """A loud sung track scores every line low, so no absolute floor: a uniformly
  bad song flags nothing, and under 5 stamps there is no ranking at all."""
  import math
  bad = [math.log(0.1)] * 20
  assert sum(1 for s in bad if s < low_conf_cut(bad)) == 0, "absolute floor crept back"
  assert low_conf_cut([0.9, 0.2, 0.5]) is None, "too few to rank"
  assert low_conf_cut([None] * 20) is None

def test_ctc_evidence_lands_only_on_adopted_lines(monkeypatch):
  """A snapped line's spans time audio the cue no longer covers, so a fill
  built from them would drift: only adopted stamps carry evidence."""
  from utasub.core import ctc_align, place
  from utasub.core.align import Cue
  spans = {0: [("a", 10.0, 10.5), ("b", 10.5, 11.0)], 1: [("c", 30.0, 30.4)]}
  cues = [Cue(10.0, 12.0, "one", "lrc"), Cue(20.0, 22.0, "two", "lrc")]
  place._attach_ctc_evidence(cues, {0}, {0: 0.9, 1: 0.3}, spans)
  assert cues[0].score == 0.9
  assert getattr(cues[1], "score", None) is None
  assert getattr(cues[1], "token_spans", None) is None

def test_ctc_spans_are_stored_relative_to_the_cue_start(monkeypatch):
  """Stored absolute, a region offset or a hand drag would desync the fill."""
  from utasub.core import ctc_align, place
  from utasub.core.align import Cue
  spans = {0: [("a", 10.2, 10.5), ("b", 10.5, 11.0)]}
  cues = [Cue(10.0, 12.0, "one", "lrc")]
  place._attach_ctc_evidence(cues, {0}, {0: 0.9}, spans)
  assert cues[0].token_spans == [("a", 0.2, 0.5), ("b", 0.5, 1.0)]

def test_plain_text_lines_get_ctc_evidence(monkeypatch):
  """No LRC timestamps, so the chain falls to coarse_fa: the global CTC pass
  still runs, every line scores and the adopted ones carry spans."""
  from utasub.core import ctc_align
  from utasub.core.align import Cue
  from utasub.core.place import AlignOpts, run_chain
  lines = _make_lines(20, spacing=5.0)
  truth = [t + _fa_offset for t, _ in lines]
  segments = _make_segments(lines)
  monkeypatch.setattr(ctc_align, "ctc_available", lambda: True)

  def _align_lines(texts, audio):
    return ({j: t for j, t in enumerate(truth)},
            {j: t + 2.0 for j, t in enumerate(truth)},
            {j: -0.5 for j in range(len(truth))},
            {j: [("a", t, t + 0.3), ("b", t + 0.3, t + 0.6)] for j, t in enumerate(truth)})
  monkeypatch.setattr(ctc_align, "align_lines", _align_lines)
  _patch_fa(monkeypatch, _fa_truth(truth))

  untimed = [(None, t) for _, t in lines]
  cues, _, meta = run_chain(untimed, segments, _fake_audio(200.0),
                            AlignOpts(use_fa=True))
  assert meta["strategy"] == "coarse_fa", meta
  placed = [c for c in cues if isinstance(c, Cue) and c.confidence != "mc"]
  assert len(placed) == 20
  assert all(c.score is not None for c in placed), "a line came back unscored"
  assert all(c.token_spans for c in placed), "no karaoke spans on an adopted line"
  assert meta["ctc_adopted"] == 20, meta
  assert abs(placed[3].start - truth[3]) < 0.05
  assert placed[3].end <= truth[3] + 2.0 + 1e-6, "ctc end did not trim the cue"
  assert placed[0].token_spans[0] == ("a", 0.0, 0.3)
