"""Unit tests for the timeline editor (offscreen), driven through the real
mouse-drag path (press→move→release), the only cue-edit surface."""
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF

from utasub.core.align import Cue
from utasub.core import session
from utasub.gui.timeline import (
  find_onsets, WaveformCanvas, TimelinePanel, UndoStack,
  DRAG_MOVE, DRAG_LEFT, DRAG_RIGHT,
)


# --- helpers ---

def _fake_cues(n=10):
  cues = []
  for i in range(n):
    s = i * 3.0
    conf = ["lrc", "fa", "coarse", "interpolated"][i % 4]
    cues.append(Cue(s, s + 2.5, f"line {i+1}", conf))
  return cues


def _fake_session(n=10):
  """n-cue session dict + synthetic sine waveform."""
  import numpy as np
  cues = _fake_cues(n)
  dur = cues[-1].end + 2.0
  sr = 16000
  audio = (np.sin(2 * np.pi * 440 * np.arange(int(dur * sr)) / sr) * 0.5).astype(np.float32)
  return {
    "schema_version": 1, "media": "fake.mkv", "chosen_index": 0, "candidates": [],
    "cues": list(cues), "song_span": (0.0, dur), "romaji": False,
    "credit_toggles": {}, "regions": [], "manual_timings": {}, "placement": {},
  }, audio


def _panel(sess_data, audio=None):
  """Build a panel the way MainWindow does: wire session/media, then load_cues."""
  panel = TimelinePanel()
  panel._media_path = sess_data["media"]
  panel._session = sess_data
  if audio is not None:
    panel.canvas.set_audio(audio, 16000)
  panel.load_cues(list(sess_data["cues"]))
  return panel


class _Ev:
  """Minimal event: mouseMoveEvent reads position().x(); release reads nothing."""
  def __init__(self, x):
    self._p = QPointF(x, 0.0)

  def position(self):
    return self._p


def _drag(canvas, idx, mode, delta_t, ripple=False, snap_suppress=True):
  """Drive the real press→move→release cue-drag path by delta_t seconds."""
  canvas.resize(1000, 200)
  canvas.duration = max(getattr(canvas, "duration", 0.0), canvas.cues[-1].end + 5.0)
  canvas.view_start = 0.0
  canvas.view_end = canvas.duration
  c = canvas.cues[idx]
  anchor = c.end if mode == DRAG_RIGHT else c.start
  x0 = canvas.time_to_x(anchor)
  x1 = canvas.time_to_x(anchor + delta_t)
  # state mousePressEvent would have set up for a cue drag
  canvas._drag_snap_suppress = snap_suppress
  canvas._drag_ripple = ripple
  canvas._drag_mode = mode
  canvas._drag_cue = idx
  canvas._drag_start_x = x0
  canvas._drag_orig_start = c.start
  canvas._drag_orig_end = c.end
  if ripple:
    canvas._drag_orig_following = [(j, canvas.cues[j].start, canvas.cues[j].end)
                                   for j in range(idx + 1, len(canvas.cues))]
  canvas.selected = idx
  canvas._push_undo_snapshot(idx)
  canvas.mouseMoveEvent(_Ev(x1))
  canvas.mouseReleaseEvent(_Ev(x1))


# --- unit tests ---

def test_find_onsets_detects_silence_to_tone_transitions():
  """Onset detection fires near each silence→tone boundary."""
  import numpy as np
  sr = 16000
  silence = np.zeros(sr, dtype=np.float32)
  tone = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.5).astype(np.float32)
  audio = np.concatenate([silence, tone, silence, tone])
  onsets = find_onsets(audio, sr=sr, gate=0.08)
  assert len(onsets) >= 2
  assert abs(onsets[0] - 1.0) < 0.15
  assert abs(onsets[1] - 3.0) < 0.15


def test_time_x_roundtrip():
  """time_to_x and x_to_time are inverse across the view."""
  canvas = WaveformCanvas()
  canvas.resize(1000, 200)
  canvas.view_start = 5.0
  canvas.view_end = 15.0
  for t in [5.0, 7.5, 10.0, 12.5, 15.0]:
    assert abs(t - canvas.x_to_time(canvas.time_to_x(t))) < 0.01


def test_onset_snap_within_window_and_alt_suppress():
  """Snap to nearest onset within 150ms; beyond it or with suppress, no snap."""
  canvas = WaveformCanvas()
  canvas.onsets = [1.0, 5.0, 10.0]
  assert canvas.snap_to_onset(1.1) == 1.0
  assert canvas.snap_to_onset(4.9) == 5.0
  assert canvas.snap_to_onset(1.2) == 1.2
  assert canvas.snap_to_onset(7.0) == 7.0
  assert canvas.snap_to_onset(1.05, suppress=True) == 1.05


def test_drag_move_shifts_cue_and_marks_dirty():
  canvas = WaveformCanvas()
  canvas.cues = _fake_cues(5)
  canvas.onsets = []
  c = canvas.cues[2]
  orig_start, orig_dur = c.start, c.end - c.start
  _drag(canvas, 2, DRAG_MOVE, 1.0)
  assert abs(canvas.cues[2].start - (orig_start + 1.0)) < 0.05
  assert abs((canvas.cues[2].end - canvas.cues[2].start) - orig_dur) < 0.05
  assert canvas.dirty


def test_drag_edges_retime_start_and_end_independently():
  canvas = WaveformCanvas()
  canvas.cues = _fake_cues(5)
  canvas.onsets = []
  orig_end = canvas.cues[2].end
  _drag(canvas, 2, DRAG_LEFT, -1.0)
  assert abs(canvas.cues[2].end - orig_end) < 0.05
  assert canvas.cues[2].start < 6.0  # moved earlier

  orig_start = canvas.cues[2].start
  _drag(canvas, 2, DRAG_RIGHT, 1.5)
  assert abs(canvas.cues[2].start - orig_start) < 0.05
  assert canvas.cues[2].end > orig_end


def test_ripple_drag_shifts_following_cues():
  canvas = WaveformCanvas()
  canvas.cues = _fake_cues(5)
  canvas.onsets = []
  orig_starts = [c.start for c in canvas.cues]
  _drag(canvas, 1, DRAG_MOVE, 2.0, ripple=True)
  delta = canvas.cues[1].start - orig_starts[1]
  for j in range(2, 5):
    assert abs(canvas.cues[j].start - (orig_starts[j] + delta)) < 0.1


def test_ripple_drag_undo_restores_every_moved_cue():
  canvas = WaveformCanvas()
  canvas.cues = _fake_cues(5)
  canvas.onsets = []
  orig = [(c.start, c.end) for c in canvas.cues]
  _drag(canvas, 1, DRAG_MOVE, 2.0, ripple=True)
  assert [(c.start, c.end) for c in canvas.cues] != orig
  canvas.do_undo()
  assert [(c.start, c.end) for c in canvas.cues] == orig


def test_backward_drag_keeps_starts_non_decreasing():
  """Dragging past a predecessor clamps instead of reordering."""
  canvas = WaveformCanvas()
  canvas.cues = _fake_cues(5)
  canvas.onsets = []
  _drag(canvas, 3, DRAG_MOVE, -20.0)
  for j in range(1, len(canvas.cues)):
    assert canvas.cues[j].start >= canvas.cues[j - 1].start


def test_undo_redo_restores_state_and_undostack_semantics():
  """Drag is undoable/redoable; UndoStack push/undo/redo/clear behave."""
  canvas = WaveformCanvas()
  canvas.cues = _fake_cues(3)
  canvas.onsets = []
  orig_start = canvas.cues[1].start
  _drag(canvas, 1, DRAG_MOVE, 1.5)
  edited_start = canvas.cues[1].start
  assert abs(edited_start - (orig_start + 1.5)) < 0.05

  canvas.do_undo()
  assert abs(canvas.cues[1].start - orig_start) < 0.05
  canvas.do_redo()
  assert abs(canvas.cues[1].start - edited_start) < 0.05

  s = UndoStack()
  assert s.undo() is None and s.redo() is None
  s.push("a")
  s.push("b")
  assert s.undo() == "b"
  assert s.undo() == "a" and s.undo() is None
  assert s.redo() == "a" and s.redo() == "b" and s.redo() is None
  s.undo()
  s.push("c")  # push after undo drops redo history
  assert s.redo() is None
  s.clear()
  assert s.undo() is None


def test_panel_builds_with_waveform_and_save_writes_edits_and_srt():
  """Save persists nudged manual edits and re-exports SRT."""
  d = Path(tempfile.mkdtemp(prefix="utasub_tl_"))
  media = d / "test.mkv"
  media.write_bytes(b"\x00")
  try:
    cues = _fake_cues(5)
    sess_data = {
      "schema_version": 1, "media": str(media.resolve()), "chosen_index": 0,
      "candidates": [], "cues": list(cues), "song_span": (0.0, 14.5),
      "romaji": False, "credit_toggles": {}, "regions": [],
      "manual_timings": {}, "placement": {},
    }
    _, audio = _fake_session(5)
    panel = _panel(sess_data, audio)
    assert len(panel.canvas.cues) == 5
    assert panel.canvas.duration > 0
    assert not panel.canvas.dirty
    panel.canvas.onsets = []
    _drag(panel.canvas, 2, DRAG_MOVE, 0.5)
    assert panel.canvas.dirty
    panel.save()
    assert not panel.canvas.dirty

    loaded = session.load(media)
    edited = {e["text"] for e in loaded["manual_edits"]["timings"]}
    assert edited, "nudged cue was not saved as a manual edit"
    srt_files = list(d.glob("*.srt"))
    assert srt_files
    assert "line 1" in srt_files[0].read_text(encoding="utf-8")
  finally:
    shutil.rmtree(d)


def test_panel_tap_sync_stamps_and_advances():
  sess, audio = _fake_session(5)
  panel = _panel(sess, audio)
  panel.canvas.selected = 1
  panel.canvas.playhead = 7.3
  panel.tap_sync()
  assert abs(panel.canvas.cues[1].start - 7.3) < 0.01
  assert panel.canvas.selected == 2
  assert panel.canvas.dirty


# --- structural edits ---

def _canvas_with_cues():
  c = WaveformCanvas()
  c.cues = [Cue(10.0, 12.0, "one", "lrc"), Cue(20.0, 22.0, "two", "lrc"),
            Cue(30.0, 32.0, "three", "lrc")]
  return c


def test_structural_edits_keep_cue_list_ordered():
  """Each op lands at the right index; order preserved."""
  c = _canvas_with_cues()
  c.insert_cue(1, Cue(15.0, 16.0, "inserted", "lrc"))
  assert [x.text for x in c.cues] == ["one", "inserted", "two", "three"]
  c.split_cue(2, 21.0)
  assert len(c.cues) == 5 and c.cues[2].end == 21.0 and c.cues[3].start == 21.0
  c.merge_cue(2)
  assert len(c.cues) == 4 and c.cues[2].text == "two two"
  c.remove_cue(1)
  assert [x.text for x in c.cues] == ["one", "two two", "three"]
  assert [x.start for x in c.cues] == sorted(x.start for x in c.cues)


def test_offset_cues_shifts_all_and_undoes():
  c = _canvas_with_cues()
  starts = [x.start for x in c.cues]
  assert c.offset_cues(1.5) == 1.5
  assert [x.start for x in c.cues] == [s + 1.5 for s in starts]
  c.do_undo()
  assert [x.start for x in c.cues] == starts


def test_offset_cues_clamps_at_zero():
  """Backward shift past the first cue's start clamps to zero."""
  c = _canvas_with_cues()
  applied = c.offset_cues(-50.0)
  assert applied == -10.0 and c.cues[0].start == 0.0


def test_resize_clamps_at_neighbour_and_snaps_when_close():
  """Right/left edge resize stops at the neighbour's edge, and snaps onto it
  from within the snap band instead of leaving a sliver gap."""
  canvas = WaveformCanvas()
  canvas.cues = _fake_cues(5)
  canvas.onsets = []
  # cue 2 spans 6.0-8.5, cue 3 starts at 9.0: a 5s stretch clamps at 9.0
  _drag(canvas, 2, DRAG_RIGHT, 5.0, snap_suppress=False)
  assert canvas.cues[2].end == 9.0
  assert canvas.cues[3].start == 9.0  # neighbour untouched

  # cue 1 starts at 3.0, cue 0 ends at 2.5: pulling left past it clamps
  _drag(canvas, 1, DRAG_LEFT, -2.0, snap_suppress=False)
  assert canvas.cues[1].start == 2.5

  # Alt bypasses the clamp, overlap allowed
  canvas.cues = _fake_cues(5)
  _drag(canvas, 2, DRAG_RIGHT, 5.0, snap_suppress=True)
  assert canvas.cues[2].end == 13.5

  # snap: stopping just short of the neighbour lands exactly on it
  canvas.cues = _fake_cues(5)
  _drag(canvas, 2, DRAG_RIGHT, 0.45, snap_suppress=False)  # 8.95, within 8px of 9.0
  assert canvas.cues[2].end == 9.0

  # Alt suppresses the neighbour snap, edge stays where dropped
  canvas.cues = _fake_cues(5)
  _drag(canvas, 2, DRAG_RIGHT, 0.45, snap_suppress=True)
  assert canvas.cues[2].end == 8.95

  # body drag snaps whichever edge lands in the band
  canvas.cues = _fake_cues(5)
  _drag(canvas, 2, DRAG_MOVE, 0.45, snap_suppress=False)  # end 8.95 -> 9.0
  assert canvas.cues[2].end == 9.0


def test_provisional_rebuild_keeps_selection_and_view():
  """Region-click refresh must not reset zoom or selection."""
  sess_data, audio = _fake_session(6)
  panel = _panel(sess_data, audio)
  panel.canvas.zoom_to_span(6.0, 9.0)
  view = (panel.canvas.view_start, panel.canvas.view_end)
  panel.canvas.selected = 2
  panel.show_provisional_cues(_fake_cues(6))
  assert panel.canvas.selected == 2
  assert (panel.canvas.view_start, panel.canvas.view_end) == view


def test_playhead_reports_region_under_it():
  """Entering/leaving a region emits its index."""
  from utasub.core.regions import Region
  canvas = WaveformCanvas()
  canvas.duration = 60.0
  canvas.regions = [Region(0.0, 10.0), Region(20.0, 30.0)]
  seen = []
  canvas.playing_region_changed.connect(seen.append)
  canvas.set_playhead(5.0)
  canvas.set_playhead(15.0)
  canvas.set_playhead(25.0)
  assert seen == [0, -1, 1]
