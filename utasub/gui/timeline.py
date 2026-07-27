"""Timeline panel: waveform canvas + cue grid + view/offset controls, save.
Re-exports the canvas/grid/util names for `from .timeline import X`."""
from dataclasses import replace

from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (
  QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollBar,
  QSplitter, QPushButton, QToolButton, QDoubleSpinBox,
)

from ..core.align import Cue
from ..core import session as sess_mod
from . import theme
from .cue_grid import CueGrid
from .waveform import WaveformCanvas
from .timeline_util import CANVAS_H
# re-exported for callers/tests that import these from .timeline
from .timeline_util import (  # noqa: F401
  romaji_for, find_onsets, UndoStack, fmt_time_mssff, parse_time, MIN_REGION_S,
  DRAG_MOVE, DRAG_LEFT, DRAG_RIGHT,
)


class TimelinePanel(QWidget):
  """Waveform canvas + cue grid + view controls, embedded in the MainWindow timeline tab."""

  def __init__(self, parent=None):
    super().__init__(parent)
    self._media_path = None   # set by the owner window once media is open
    self._session = {}        # session dict the owner keeps in sync
    self._provisional = False

    # canvas
    self.canvas = WaveformCanvas()

    # cue grid
    self.grid = CueGrid()

    # empty-state hint overlay on grid viewport
    self._hint_label = QLabel(
      "No cues yet — assign lyrics per region, then Align all + Export",
      self.grid.viewport())
    self._hint_label.setAlignment(Qt.AlignCenter)
    self._hint_label.setWordWrap(True)
    self._hint_label.setStyleSheet(f"color: {theme.HINT_GRAY}; font-style: italic;")
    self._hint_label.hide()
    self.grid.viewport().installEventFilter(self)

    # view-control row: Fit | go-to-playhead | Follow (small flat buttons)
    self._syncing_scroll = False  # guard against scrollbar<->canvas feedback
    controls = QHBoxLayout()
    controls.setContentsMargins(2, 0, 2, 0)
    controls.setSpacing(4)
    flat_css = ("QAbstractButton { padding: 2px 8px; border: 1px solid #444;"
                " border-radius: 3px; background: #33333a; }"
                " QAbstractButton:hover { background: #3d3d47; }"
                " QAbstractButton:checked { background: #46628f; }")
    fit_btn = QPushButton("Fit")
    fit_btn.setFixedHeight(20)
    fit_btn.setToolTip("Zoom to show entire timeline")
    fit_btn.clicked.connect(self.canvas.fit_all)
    goto_btn = QPushButton("⌖")  # crosshair: center on playhead
    goto_btn.setFixedSize(24, 20)
    goto_btn.setToolTip("Go to playhead")
    goto_btn.clicked.connect(self._go_to_playhead)
    self.follow_btn = QToolButton()
    self.follow_btn.setText("Follow")
    self.follow_btn.setCheckable(True)
    self.follow_btn.setChecked(True)
    self.follow_btn.setFixedHeight(20)
    self.follow_btn.setToolTip("Follow playhead during playback")
    self.follow_btn.toggled.connect(self._on_follow_toggled)
    for b in (fit_btn, goto_btn, self.follow_btn):
      b.setStyleSheet(flat_css)
    controls.addWidget(fit_btn)
    controls.addWidget(goto_btn)
    controls.addWidget(self.follow_btn)
    controls.addStretch(1)

    # cue offset: step box + -/+ shifting every cue. Edits stored timings,
    # unlike the toolbar SRT offset which applies at export only.
    self.offset_spin = QDoubleSpinBox()
    self.offset_spin.setRange(-600.0, 600.0)
    self.offset_spin.setDecimals(2)
    self.offset_spin.setSingleStep(0.05)
    self.offset_spin.setValue(0.10)
    self.offset_spin.setSuffix(" s")
    self.offset_spin.setFixedSize(70, 20)
    self.offset_spin.setToolTip("Shift step used by the -/+ buttons")
    controls.addWidget(QLabel("Shift all cues:"))
    controls.addWidget(self.offset_spin)
    for sign, glyph in ((-1, "-"), (1, "+")):
      b = QPushButton(glyph)
      b.setFixedSize(22, 20)
      b.setStyleSheet(flat_css)
      b.setToolTip(f"Shift every cue {'earlier' if sign < 0 else 'later'} by the step")
      b.clicked.connect(lambda _=False, s=sign:
                        self.offset_cues(s * self.offset_spin.value()))
      controls.addWidget(b)

    # horizontal scrollbar (centi-second resolution)
    self.scrollbar = QScrollBar(Qt.Horizontal)
    self.scrollbar.valueChanged.connect(self._on_scroll)

    # top block: canvas + controls + scrollbar stacked
    top = QWidget()
    top_layout = QVBoxLayout(top)
    top_layout.setContentsMargins(0, 0, 0, 0)
    top_layout.setSpacing(2)
    top_layout.addWidget(self.canvas, 1)
    top_layout.addLayout(controls)
    top_layout.addWidget(self.scrollbar)

    # splitter: canvas block on top, grid on bottom
    splitter = QSplitter(Qt.Vertical)
    splitter.addWidget(top)
    splitter.addWidget(self.grid)
    splitter.setStretchFactor(0, 0)  # canvas keeps its height...
    splitter.setStretchFactor(1, 1)  # ...the grid absorbs extra window height
    splitter.setSizes([CANVAS_H, 400])

    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(splitter, 1)

    # canvas view -> scrollbar sync (guarded)
    self.canvas.view_changed.connect(self._sync_scrollbar_from_canvas)

    self.canvas.fit_all()
    self.grid.load_cues(self.canvas.cues)

    # bidirectional sync: canvas <-> grid
    self.canvas.selection_changed.connect(self._on_canvas_select)
    self.canvas.cue_changed.connect(self._on_canvas_cue_changed)
    self.canvas.playing_cue_changed.connect(self.grid.set_playing_row)
    self.grid.selection_sync.connect(self._on_grid_select)
    self.grid.cue_edited.connect(self._on_grid_edit)

    self._update_hint()

  def eventFilter(self, obj, ev):
    if obj is self.grid.viewport() and ev.type() == QEvent.Resize:
      self._update_hint()
    return super().eventFilter(obj, ev)

  def _update_hint(self):
    """Show/size the empty-state hint over the grid viewport."""
    empty = self.grid.rowCount() == 0 and not self._provisional
    self._hint_label.setVisible(empty)
    if empty:
      vp = self.grid.viewport()
      self._hint_label.setGeometry(0, 0, vp.width(), vp.height())

  # --- view controls + scrollbar sync ---

  def _sync_scrollbar_from_canvas(self):
    """Reflect canvas view (duration/visible span) onto scrollbar. Guarded."""
    if self._syncing_scroll:
      return
    self._syncing_scroll = True
    cv = self.canvas
    total = int(round(cv.duration * 100))
    span = int(round((cv.view_end - cv.view_start) * 100))
    sb = self.scrollbar
    if total <= 0 or span <= 0 or span >= total:
      sb.setRange(0, 0)
      sb.setPageStep(max(span, 1))
      sb.setValue(0)
      sb.setEnabled(False)
    else:
      sb.setEnabled(True)
      sb.setPageStep(span)
      sb.setSingleStep(max(1, span // 10))
      sb.setRange(0, total - span)
      sb.setValue(int(round(cv.view_start * 100)))
    self._syncing_scroll = False

  def _on_scroll(self, value):
    """Scrollbar drag pans the canvas, keeping span. Guarded."""
    if self._syncing_scroll:
      return
    cv = self.canvas
    span = cv.view_end - cv.view_start
    self._syncing_scroll = True
    cv.view_start = value / 100.0
    cv.view_end = cv.view_start + span
    cv._clamp_view()
    cv.update()
    self._syncing_scroll = False

  def _go_to_playhead(self):
    """Center view on playhead, keeping current zoom span."""
    cv = self.canvas
    span = cv.view_end - cv.view_start
    cv.view_start = max(0.0, cv.playhead - span / 2)
    cv.view_end = cv.view_start + span
    cv._clamp_view()
    cv.view_changed.emit()
    cv.update()

  def _on_follow_toggled(self, checked):
    self.canvas.follow = checked
    self.grid.follow_playhead = checked

  def show_provisional_cues(self, cues):
    """Display provisional (pre-export) cues dimmed; no dirty/undo, save() no-ops.
    A later load_cues() clears the provisional flag and replaces display."""
    self._provisional = True
    self.canvas.provisional = True
    # a rebuild triggered by a region click must not throw away the user's
    # selection or zoom; only the very first preview fits the whole timeline
    prev_sel = self.canvas.selected
    prev_view = (self.canvas.view_start, self.canvas.view_end)
    had_cues = bool(self.canvas.cues)
    song_cues = []
    for c in cues:
      if isinstance(c, Cue):
        song_cues.append(c)
      else:
        song_cues.append(Cue(c[0], c[1], c[2],
                             c[3] if len(c) > 3 else "provisional"))
    self.canvas.cues = song_cues
    if song_cues:
      self.canvas.duration = max(
        max(c.end for c in song_cues) + 2.0, self.canvas.duration)
    if had_cues:
      self.canvas.view_start, self.canvas.view_end = prev_view
      self.canvas._clamp_view()
      self.canvas.view_changed.emit()
      self.canvas.update()
    else:
      self.canvas.fit_all()
    self.grid.set_provisional(True)
    self.grid.load_cues(self.canvas.cues)
    self.canvas.selected = prev_sel if 0 <= prev_sel < len(song_cues) else -1
    self.grid.select_row(self.canvas.selected)
    self._update_hint()

  def load_cues(self, cues, regions=None, edits=None):
    """Load cues into canvas+grid (e.g. after finalize).
    cues: list of Cue or (start, end, text) tuples.
    edits: manual timings and deletions to re-apply over this fresh placement.
    Incoming cues become the baseline first, so re-applied edits still read as
    edits at the next save and outlive a second align."""
    from ..core.session import apply_manual_edits, edit_key
    self._provisional = False
    self.canvas.provisional = False
    self.grid.set_provisional(False)
    self._baseline = [(edit_key(c[2]), c[0], c[1]) for c in cues]
    cues = apply_manual_edits(cues, edits or self._manual_edits())
    song_cues = [c if isinstance(c, Cue)
                 else Cue(c[0], c[1], c[2], c[3] if len(c) > 3 else None)
                 for c in cues]
    self.canvas.cues = song_cues
    if song_cues:
      self.canvas.duration = max(max(c.end for c in song_cues) + 2.0, self.canvas.duration)
    if regions is not None:
      self.canvas.regions = regions
    self.canvas.undo_stack.clear()
    self.canvas.clear_dirty()
    self.canvas.fit_all()
    self.grid.load_cues(self.canvas.cues)
    self._update_hint()

  # --- bidirectional sync ---

  def _on_canvas_select(self, idx):
    self.grid.select_row(idx)

  def _on_canvas_cue_changed(self):
    """Canvas drag/edit finished: update affected grid row."""
    idx = self.canvas.selected
    if 0 <= idx < len(self.canvas.cues):
      self.grid.update_row(idx)

  def _on_grid_select(self, row):
    if 0 <= row < len(self.canvas.cues):
      self.canvas.selected = row
      self.canvas.selection_changed.emit(row)
      self.canvas.update()

  def _on_grid_edit(self, idx, new_cue):
    """Grid inline edit: apply to canvas cues + push undo."""
    if self._provisional or idx < 0 or idx >= len(self.canvas.cues):
      return
    old = self.canvas.cues[idx]
    old_copy = replace(old)
    self.canvas.cues[idx] = new_cue
    self.canvas.undo_stack.push(('timing', idx, old_copy, replace(new_cue)))
    self.canvas._mark_dirty()

  # --- structural ops ---

  def _grid_editing(self):
    """True while a grid cell is in inline edit, so Del/Ins keep editing text."""
    from PySide6.QtWidgets import QAbstractItemView
    return self.grid.state() == QAbstractItemView.EditingState

  def do_insert(self):
    """Insert new cue after selected (or at playhead)."""
    if self._grid_editing():
      return
    idx = self.canvas.selected
    cues = self.canvas.cues
    if idx >= 0 and idx < len(cues):
      c = cues[idx]
      # split gap to next cue
      if idx + 1 < len(cues):
        gap_end = cues[idx + 1].start
      else:
        gap_end = c.end + 2.0
      new_start = c.end
      new_end = min(c.end + (gap_end - c.end) / 2, gap_end)
      new_end = max(new_start + 0.1, new_end)
      insert_at = idx + 1
    else:
      # no selection: insert at playhead
      new_start = self.canvas.playhead
      new_end = new_start + 2.0
      insert_at = len(cues)
      for i, c in enumerate(cues):
        if c.start > new_start:
          insert_at = i
          break
    new_cue = Cue(round(new_start, 3), round(new_end, 3), "", None)
    self.canvas.insert_cue(insert_at, new_cue)
    self.grid.rebuild()
    self._update_hint()
    self.canvas.selected = insert_at
    self.canvas.selection_changed.emit(insert_at)
    self.grid.select_row(insert_at)

  def _collect_edits(self, cues):
    """Manual edits to persist: recorded deletions, plus every cue whose timing
    differs from the baseline placement. Diffing against the baseline leaves
    untouched lines free for the next align to improve."""
    from ..core.session import _edit_record, edit_key
    edits = self._manual_edits()
    base = {}
    for key, s, e in (getattr(self, "_baseline", None) or []):
      base.setdefault(key, []).append((s, e))
    kept = {(t["text"], round(t["at"], 1)): t for t in edits["timings"]}
    for c in cues:
      key = edit_key(c.text)
      near = base.get(key)
      if near:
        s, e = min(near, key=lambda p: abs(p[0] - c.start))
        if abs(s - c.start) < 0.005 and abs(e - c.end) < 0.005:
          continue  # untouched: leave it to the aligner next time
      rec = _edit_record((c.start, c.end, c.text))
      kept[(rec["text"], round(rec["at"], 1))] = rec
    edits["timings"] = sorted(kept.values(), key=lambda t: t["at"])
    return edits

  def _manual_edits(self):
    """Manual edits carried by the session, in re-attachable form."""
    e = (self._session or {}).get("manual_edits") or {}
    return {"timings": list(e.get("timings") or []),
            "deleted": list(e.get("deleted") or [])}

  def reset_manual_edits(self):
    """Forget every hand timing and deletion so the next Align places all lines
    fresh. Baseline is re-pointed at the current cues, else a save before that
    Align would diff against the old placement and rewrite every line as an edit."""
    from ..core.session import edit_key
    if self._session is None:
      return False
    self._session["manual_edits"] = {"timings": [], "deleted": []}
    self._session["manual_timings"] = {}
    self._baseline = [(edit_key(c.text), c.start, c.end) for c in self.canvas.cues]
    return True

  def do_delete(self):
    if self._grid_editing():
      return
    idx = self.canvas.selected
    if idx < 0 or idx >= len(self.canvas.cues):
      return
    # remember it by text and position: a rebuild would otherwise put it back
    from ..core.session import _edit_record
    c = self.canvas.cues[idx]
    edits = self._manual_edits()
    edits["deleted"].append(_edit_record((c.start, c.end, c.text)))
    self._session["manual_edits"] = edits
    self.canvas.remove_cue(idx)
    self.grid.rebuild()
    self._update_hint()
    new_sel = min(idx, len(self.canvas.cues) - 1)
    if new_sel >= 0:
      self.canvas.selected = new_sel
      self.canvas.selection_changed.emit(new_sel)
      self.grid.select_row(new_sel)

  def do_split(self):
    idx = self.canvas.selected
    if idx < 0 or idx >= len(self.canvas.cues):
      return
    self.canvas.split_cue(idx, self.canvas.playhead)
    self.grid.rebuild()

  def do_merge(self):
    idx = self.canvas.selected
    if idx < 0 or idx + 1 >= len(self.canvas.cues):
      return
    self.canvas.merge_cue(idx)
    self.grid.rebuild()

  def offset_cues(self, delta):
    """Shift every cue by delta seconds (one undo step). No-op while provisional."""
    if self._provisional:
      return
    applied = self.canvas.offset_cues(delta)
    if not applied:
      return
    self.grid.rebuild()
    print(f"  cue offset {applied:+.2f}s applied to all cues")

  def do_undo(self):
    self.canvas.do_undo()
    self.grid.rebuild()

  def do_redo(self):
    self.canvas.do_redo()
    self.grid.rebuild()

  def tap_sync(self):
    """Stamp selected/next cue start at playhead, advance selection.
    Grid stays synced via canvas cue_changed/selection_changed signals."""
    canvas = self.canvas
    idx = canvas.selected
    if idx < 0:
      idx = 0
    if idx >= len(canvas.cues):
      return
    t = canvas.playhead
    c = canvas.cues[idx]
    old = replace(c)
    dur = c.end - c.start
    canvas.cues[idx] = replace(c, start=t, end=t + dur)
    canvas._enforce_non_decreasing(idx)
    canvas.undo_stack.push(('timing', idx, old, replace(canvas.cues[idx])))
    if idx + 1 < len(canvas.cues):
      canvas.selected = idx + 1
      canvas.selection_changed.emit(idx + 1)
    canvas._mark_dirty()

  # --- save ---

  def save(self, export=True):
    """Save manual edits + cues to session; re-export SRT when export=True.
    Returns True on success."""
    if self._provisional:
      return False  # provisional display is not persisted
    if not self._media_path:
      return False
    cues = self.canvas.cues
    edits = self._collect_edits(cues)
    self._session["manual_edits"] = edits

    self._session["cues"] = list(cues)

    sess_mod.save(
      self._media_path,
      candidates=self._session.get("candidates", []),
      chosen_index=self._session.get("chosen_index"),
      cues=list(cues),
      song_span=self._session.get("song_span"),
      romaji=self._session.get("romaji", True),
      credit_toggles=self._session.get("credit_toggles", {}),
      regions=self._session.get("regions", []),
      manual_edits=edits,
      placement=self._session.get("placement", {}),
      region_metas=self._session.get("region_metas", []),
      offset=self._session.get("offset"),
      lead=self._session.get("lead"),
      region_choices=self._session.get("region_choices"),
    )
    if export:
      sess_mod.reexport(self._media_path, self._session)
    self.canvas.clear_dirty()
    return True
