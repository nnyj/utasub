"""Timeline panel: waveform canvas + cue grid + view/offset controls, save.
Re-exports the canvas/grid/util names for `from .timeline import X`."""
from dataclasses import replace

from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (
  QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollBar,
  QSplitter, QPushButton, QToolButton, QDoubleSpinBox, QMessageBox,
)

from ..core.align import Cue
from ..core import session as sess_mod
from . import theme
from .cue_grid import CueGrid
from .waveform import WaveformCanvas, as_cue
from .timeline_util import (
  CANVAS_H, EDIT_EPS_S, TEXT_NEAR_S, AUTO_ZOOM_PX, NUDGE_S,
  NUDGE_COARSE_S,
)
# re-exported for callers/tests that import these from .timeline
from .timeline_util import (  # noqa: F401
  romaji_for, find_onsets, UndoStack, fmt_time_mssff, parse_time, MIN_REGION_S,
  DRAG_MOVE, DRAG_LEFT, DRAG_RIGHT,
)

def occurrence_keys(cues):
  """(normalized text, nth repeat) per cue: the identity a manual timing record
  keeps across a re-align. Text alone collides on a repeated chorus line."""
  from ..core.session import edit_key
  seen, out = {}, []
  for c in cues:
    k = edit_key(c.text)
    out.append((k, seen.get(k, 0)))
    seen[k] = seen.get(k, 0) + 1
  return out

def rebind_timings(records, cues, keys):
  """Existing timing records re-keyed onto the cue each still describes.
  A record whose line is on screen but whose timing matches none of that line's
  cues was superseded by a later drag of the same cue: keeping it lets the next
  load claim the *next* repeat of the line and mis-time it, so it is dropped.
  A record whose line has no cue at all survives, a later align may place it."""
  by_text = {}
  for i, (text, _) in enumerate(keys):
    by_text.setdefault(text, []).append(i)
  used, out = set(), {}
  for n, rec in enumerate(records):
    same = by_text.get(rec.get("text", ""), [])
    hit = next((i for i in same if i not in used
                and abs(cues[i].start - rec.get("start", -1)) < EDIT_EPS_S
                and abs(cues[i].end - rec.get("end", -1)) < EDIT_EPS_S), None)
    if hit is not None:
      used.add(hit)
      out[keys[hit]] = rec
    elif not same:
      out[("\0unmatched", n)] = rec  # no cue of this text: keep, key can't clash
  return out

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
    help_btn = QToolButton()
    help_btn.setText("?")
    help_btn.setFixedSize(24, 20)
    help_btn.setToolTip("Timeline controls")
    help_btn.clicked.connect(lambda: QMessageBox.information(
      self, "Timeline controls",
      "Drag a cue to move; drag cue edges to resize.\n"
      "Shift+drag: ripple following cues. Alt: disable snapping.\n"
      "Shift+wheel: pan. Drag region flags in the ruler to edit bounds."))
    controls.addWidget(help_btn)
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

    # splitter: canvas block on top, grid on bottom. The waveform block keeps
    # CANVAS_H at any window size (stretch 0), a taller window grows the grid.
    self._splitter = QSplitter(Qt.Vertical)
    self._splitter.addWidget(top)
    self._splitter.addWidget(self.grid)
    self._splitter.setStretchFactor(0, 0)
    self._splitter.setStretchFactor(1, 1)
    self._splitter.setSizes([CANVAS_H, 400])
    self._split_applied = False

    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(self._splitter, 1)

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

  def showEvent(self, ev):
    """Give the grid the rest of the height once it is known; later drags stick."""
    super().showEvent(ev)
    h = self.height()
    if not self._split_applied and h > CANVAS_H:
      self._split_applied = True
      self._splitter.setSizes([CANVAS_H, h - CANVAS_H])

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
    """Display provisional cues dimmed; saving preserves only region metadata.
    A later load_cues() clears the provisional flag and replaces display."""
    self._provisional = True
    self.canvas.provisional = True
    # a rebuild triggered by a region click must not throw away the user's
    # selection or zoom; only the very first preview fits the whole timeline
    prev_sel = self.canvas.selected
    prev_view = (self.canvas.view_start, self.canvas.view_end)
    had_cues = bool(self.canvas.cues)
    song_cues = [as_cue(c, "provisional") for c in cues]
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
    edits = dict(edits or self._manual_edits())
    base_cues = [as_cue(c) for c in cues]
    # prune superseded records before applying, else a stale one claims the next
    # repeat of its line and re-times it (repeated-chorus corruption)
    edits["timings"] = sorted(
      rebind_timings(edits.get("timings") or [], base_cues,
                     occurrence_keys(base_cues)).values(),
      key=lambda t: t["at"])
    if self._session is not None:
      self._session["manual_edits"] = edits
    song_cues = [as_cue(c) for c in apply_manual_edits(base_cues, edits)]
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
      self._scroll_into_view(row)

  def _scroll_into_view(self, idx):
    """Row click: keep the zoom, pan only when the cue is off screen."""
    cv = self.canvas
    c = cv.cues[idx]
    if c.start < cv.view_start or c.end > cv.view_end:
      span = cv.view_end - cv.view_start
      cv.view_start = max(0.0, (c.start + c.end) / 2 - span / 2)
      cv.view_end = cv.view_start + span
      cv._clamp_view()
      cv.view_changed.emit()
    cv.update()

  def _reveal(self, idx):
    """Keep the selected cue workable: zoom the view onto it when it is too
    narrow to grab or sits outside the visible span."""
    cv = self.canvas
    c = cv.cues[idx]
    narrow = cv.time_to_x(c.end) - cv.time_to_x(c.start) < AUTO_ZOOM_PX
    if narrow or c.start < cv.view_start or c.end > cv.view_end:
      cv.zoom_to_span(max(0.0, c.start - 2.0), c.end + 2.0)
    cv.update()

  # --- keyboard cue navigation / nudge ---

  def select_cue(self, idx):
    """Select cue idx from anywhere, syncing canvas, grid and view."""
    if not (0 <= idx < len(self.canvas.cues)):
      return
    self.canvas.selected = idx
    self.canvas.selection_changed.emit(idx)
    self.grid.select_row(idx)
    self._reveal(idx)

  def step_cue(self, delta):
    """N / P: move the selection one cue on, wrapping at neither end."""
    if self._grid_editing() or not self.canvas.cues:
      return
    cur = self.canvas.selected
    self.select_cue(0 if cur < 0 else
                    min(max(cur + delta, 0), len(self.canvas.cues) - 1))

  def low_conf_rows(self):
    """Row indices worth a second listen (see ctc_align.low_conf_cut), in order.
    Empty when nothing carries a score (an LRC-as-is or coarse pass)."""
    from ..core.ctc_align import low_conf_cut
    scores = [getattr(c, "score", None) for c in self.canvas.cues]
    cut = low_conf_cut(scores)
    if cut is None:
      return []
    return [i for i, s in enumerate(scores) if s is not None and s < cut]

  def step_low_conf(self, delta):
    """Alt+N / Alt+P: jump to the next/previous low-confidence cue, wrapping so
    a pass over the cues runs round the song without hunting for the first one."""
    rows = self.low_conf_rows()
    if not rows:
      print("  no low-confidence cues"
            if any(getattr(c, "score", None) is not None for c in self.canvas.cues)
            else "  no CTC scores on these cues, nothing to check")
      return
    cur = self.canvas.selected
    if delta > 0:
      later = [i for i in rows if i > cur]
      self.select_cue(later[0] if later else rows[0])
    else:
      earlier = [i for i in rows if i < cur]
      self.select_cue(earlier[-1] if earlier else rows[-1])

  def nudge_cue(self, steps, edge="start", coarse=False):
    """Arrow keys: shift the selected cue's start (whole block) or just its end."""
    if self._grid_editing():
      return
    idx = self.canvas.selected
    if not (0 <= idx < len(self.canvas.cues)) or self._provisional:
      return
    step = steps * (NUDGE_COARSE_S if coarse else NUDGE_S)
    c = self.canvas.cues[idx]
    if edge == "end":
      self.canvas.retime_cue(idx, end=c.end + step)
    else:
      self.canvas.retime_cue(idx, start=c.start + step, end=c.end + step)
    self.grid.update_row(idx)

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
    or text differs from the baseline placement. Diffing against the baseline leaves
    untouched lines free for the next align to improve. Records are keyed by
    (text, occurrence) so re-dragging a repeated line replaces its own record."""
    from ..core.session import _edit_record
    edits = self._manual_edits()
    baseline = getattr(self, "_baseline", None)
    if not baseline:
      # nothing was loaded this run, so every cue would read as hand-inserted
      # and get replayed verbatim over every future align
      return edits
    base_by_text = {}
    for n, (key, _s, _e) in enumerate(baseline):
      base_by_text.setdefault(key, []).append(n)
    keys = occurrence_keys(cues)
    kept = rebind_timings(edits["timings"], cues, keys)
    matched, unmatched = {}, []
    for i, (c, key) in enumerate(zip(cues, keys)):
      idxs = base_by_text.get(key[0]) or []
      if key[1] < len(idxs):
        matched[i] = min(idxs, key=lambda n: abs(baseline[n][1] - c.start))
      else:
        unmatched.append(i)
    used = set(matched.values())
    texts, added = [], []
    for i in unmatched:
      # this placement never produced this line: either the text was retyped
      # over a line it did produce (a baseline cue at the same time is still
      # free) or the line was hand-inserted
      c = cues[i]
      free = [n for n in range(len(baseline)) if n not in used]
      near = min(free, key=lambda n: abs(baseline[n][1] - c.start), default=None)
      if near is not None and abs(baseline[near][1] - c.start) <= TEXT_NEAR_S:
        used.add(near)
        rec = {"text": baseline[near][0], "at": round(baseline[near][1], 3),
               "text_to": c.text}
        if (abs(baseline[near][1] - c.start) >= EDIT_EPS_S
            or abs(baseline[near][2] - c.end) >= EDIT_EPS_S):
          # the record carries the timing too: a timing record would key on the
          # typed text and claim some other cue on the next align
          rec.update(start=round(c.start, 3), end=round(c.end, 3))
        texts.append(rec)
      else:
        added.append({**_edit_record(c), "text_raw": c.text})
    for i, n in matched.items():
      c = cues[i]
      _key, s, e = baseline[n]
      if abs(s - c.start) < EDIT_EPS_S and abs(e - c.end) < EDIT_EPS_S:
        continue  # untouched: leave it to the aligner next time
      kept[keys[i]] = _edit_record((c.start, c.end, c.text))
    edits["timings"] = sorted(kept.values(), key=lambda t: t["at"])
    edits["added"] = sorted(added, key=lambda t: t["at"])
    edits["texts"] = sorted(texts, key=lambda t: t["at"])
    return edits

  def _manual_edits(self):
    """Manual edits carried by the session, in re-attachable form."""
    e = (self._session or {}).get("manual_edits") or {}
    from ..core.session import EDIT_KINDS
    return {k: list(e.get(k) or []) for k in EDIT_KINDS}

  def reset_manual_edits(self):
    """Forget every hand timing and deletion so the next Align places all lines
    fresh. Baseline is re-pointed at the current cues, else a save before that
    Align would diff against the old placement and rewrite every line as an edit."""
    from ..core.session import edit_key
    if self._session is None:
      return False
    from ..core.session import empty_edits
    self._session["manual_edits"] = empty_edits()
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
    One undo entry covers the stamped cue and any it shoved."""
    canvas = self.canvas
    idx = max(canvas.selected, 0)
    if idx >= len(canvas.cues):
      return
    t = canvas.playhead
    c = canvas.cues[idx]
    canvas.retime_cue(idx, start=t, end=t + (c.end - c.start))
    self.grid.rebuild()
    if idx + 1 < len(canvas.cues):
      canvas.selected = idx + 1
      canvas.selection_changed.emit(idx + 1)
      self.grid.select_row(idx + 1)

  # --- save ---

  def _sync_session_cues(self):
    """Push current cues + manual edits into the session dict. Returns
    (cues, edits), or None when the timeline is provisional or has no media."""
    if self._provisional or not self._media_path:
      return None
    cues = list(self.canvas.cues)
    edits = self._collect_edits(cues)
    self._session["manual_edits"] = edits
    self._session["cues"] = cues
    return cues, edits

  def export_srt(self):
    """Write SRT from the current cues. Session file untouched."""
    if self._sync_session_cues() is None:
      return False
    sess_mod.reexport(self._media_path, self._session)
    return True

  def save(self, export=True):
    """Save manual edits + cues to session; re-export SRT when export=True.
    Returns True on success."""
    if self._provisional and self._media_path and not export:
      saved = sess_mod.load(self._media_path) or {}
      synced = saved.get("cues", []), self._session.get("manual_edits", {})
    else:
      synced = self._sync_session_cues()
    if synced is None:
      return False
    cues, edits = synced

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
      ass_style=self._session.get("ass_style"),
      translations=self._session.get("translations"),
    )
    if export:
      sess_mod.reexport(self._media_path, self._session)
    self.canvas.clear_dirty()
    return True
