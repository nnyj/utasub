"""Region dock for MainWindow: row population, selection, context-menu edits,
boundary handling. Mixed into MainWindow, so it uses its state directly."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (
  QDockWidget, QTreeWidget, QTreeWidgetItem, QHeaderView, QMenu, QMessageBox,
  QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QLabel,
)

from ..core.align import SIMILARITY_THRESHOLD
from . import theme

STATUS_HINTS = {
  "setlist": "assigned from a setlist",
  "found": "confident lyric match",
  "ambiguous": "weak match, check the picker",
  "none": "no usable candidate found",
}

STATUS_HINT = (
  "Match status:\n"
  "  star - assigned from a setlist\n"
  "  filled dot - confident lyric match\n"
  "  hollow dot - weak match, check the picker\n"
  "  dash - no usable candidate found\n"
  "  blank - no search run yet")

TITLE_COL_MAX = 150  # px cap on the region Title column, keeps Strategy in view

STRATEGY_HINT = (
  "Alignment strategy (post-align):\n"
  "  pending - assigned, not yet aligned\n"
  "  lrc_trust - trusted LRC timestamps (single offset)\n"
  "  lrc_anchor - LRC intervals, FA-anchored drift (live tempo)\n"
  "  lrc_force - forced LRC placement, refine by hand\n"
  "  coarse_fa / coarse_envelope - forced-align / envelope estimate\n"
  "  low_confidence_asr - weak match, kept ASR\n"
  "  unassigned - no lyrics, kept ASR")

class _NoAutofitHeader(QHeaderView):
  """Double-click-to-autofit blows the Title column up and pushes Strategy out
  of the dock, so swallow the event."""

  def mouseDoubleClickEvent(self, ev):
    ev.accept()

class RegionListMixin:
  """Region dock construction + all region-list behaviour."""

  def _build_region_dock(self):
    self._region_dock = QDockWidget("Regions", self)
    self._region_list = QTreeWidget()
    self._region_list.setMaximumWidth(360)
    self._region_list.setColumnCount(5)
    self._region_list.setHeaderLabels(["#", "Time", "St", "Title", "Strategy"])
    self._region_list.headerItem().setToolTip(2, STATUS_HINT)
    self._region_list.headerItem().setToolTip(4, STRATEGY_HINT)
    self._region_list.setRootIsDecorated(False)
    self._region_list.setUniformRowHeights(True)
    self._region_list.setWordWrap(False)
    self._region_list.setTextElideMode(Qt.ElideRight)
    header = _NoAutofitHeader(Qt.Horizontal, self._region_list)
    header.setSectionResizeMode(QHeaderView.Interactive)
    header.setStretchLastSection(True)
    self._region_list.setHeader(header)
    self._region_list.currentItemChanged.connect(self._on_region_item_changed)
    self._region_list.setContextMenuPolicy(Qt.CustomContextMenu)
    self._region_list.customContextMenuRequested.connect(self._on_region_context_menu)
    self._region_dock.setWidget(self._region_list)
    self.addDockWidget(Qt.LeftDockWidgetArea, self._region_dock)
    self._playing_region = -1  # row tinted to follow the playhead

  # --- selection ---

  def _on_region_item_changed(self, current, previous):
    row = self._region_list.indexOfTopLevelItem(current) if current else -1
    self._on_region_select(row)

  def _select_region_row(self, row):
    item = self._region_list.topLevelItem(row)
    if item is not None:
      self._region_list.setCurrentItem(item)

  def _populate_region_list(self):
    """Rebuild rows, keeping the active region selected. Signals stay blocked:
    clear() would emit currentItemChanged(None) and drop _active_region."""
    if not self._regions:
      return
    from ..core.regions import fmt_time
    from ..core.romanize import romanize_suffix
    self._region_list.blockSignals(True)
    self._region_list.clear()
    for i, r in enumerate(self._regions):
      status = self._region_status(i)
      title = self._region_title(i)
      rom = romanize_suffix(title)
      strategy = self._region_strategy(i)
      item = QTreeWidgetItem([
        f"{i+1:02d}", f"{fmt_time(r.start)}-{fmt_time(r.end)}",
        "", f"{title}{rom}", strategy])
      item.setIcon(2, theme.status_icon(status))
      item.setToolTip(2, STATUS_HINTS.get(status, "no search run yet"))
      if strategy == "pending":
        item.setForeground(4, QBrush(theme.GRAY))
      item.setToolTip(3, f"{title}{rom}")  # full title, column is capped
      if self._region_suspect(i):
        # a weak or near-tied auto-assignment: the lyrics may be the wrong
        # song's entirely, which no amount of nudging in the timeline fixes
        item.setForeground(3, QBrush(theme.CONF_LOW))
        runner = getattr(r, "runner_up", "")
        item.setToolTip(3, f"{title}{rom}\nweak match, check the picker"
                        + (f"\nrunner-up: {runner}" if runner else ""))
      self._region_list.addTopLevelItem(item)
    for c in range(self._region_list.columnCount()):
      self._region_list.resizeColumnToContents(c)
    self._region_list.setColumnWidth(
      3, min(self._region_list.columnWidth(3), TITLE_COL_MAX))
    if self._active_region is not None:
      self._select_region_row(self._active_region)
    self._paint_playing_region()  # rebuilt items carry no tint
    self._region_list.blockSignals(False)

  def _on_playing_region_changed(self, idx):
    """Playhead entered/left a region: tint that row. Selection is untouched,
    so playback never yanks the picker or the zoom to another region."""
    if idx == self._playing_region:
      return
    self._playing_region = idx
    self._paint_playing_region()
    item = self._region_list.topLevelItem(idx) if idx >= 0 else None
    if item is not None:
      self._region_list.scrollToItem(item)

  def _paint_playing_region(self):
    """Apply the playhead tint to the current rows."""
    playing = QBrush(theme.PLAYING_ROW)
    clear = QBrush(Qt.NoBrush)
    for row in range(self._region_list.topLevelItemCount()):
      item = self._region_list.topLevelItem(row)
      brush = playing if row == self._playing_region else clear
      for col in range(self._region_list.columnCount()):
        item.setBackground(col, brush)

  def _region_strategy(self, idx):
    """Strategy label from region_metas (post-align). 'pending' when the region
    is assigned but not yet aligned; blank when nothing is assigned."""
    for m in self._region_metas:
      if m.get("region_idx") == idx:
        return m.get("strategy", "")
    if idx in self._region_chosen:
      return "pending"
    return ""

  def _region_suspect(self, idx):
    """True when the DP flagged this region's auto-assignment. A hand pick in
    the picker clears it: the user has already made the call the flag asks for."""
    chosen = self._region_chosen.get(idx)
    if chosen is not None and getattr(chosen[1], "user_picked", False):
      return False
    return bool(getattr(self._regions[idx], "suspect", False))

  def _region_status(self, idx):
    chosen = self._region_chosen.get(idx)
    if chosen is not None:
      src = getattr(chosen[1], "source", "")
      if "Setlist" in src:
        return "setlist"
    scored = self._region_scored.get(idx, [])
    if not scored:
      return None
    best_score = scored[0][0] if scored else 0
    if best_score >= SIMILARITY_THRESHOLD:
      return "found"
    if best_score >= SIMILARITY_THRESHOLD * 0.7:
      return "ambiguous"
    return "none"

  def _region_title(self, idx):
    chosen = self._region_chosen.get(idx)
    if chosen is not None:
      return chosen[1].title
    scored = self._region_scored.get(idx, [])
    if scored and scored[0][0] >= SIMILARITY_THRESHOLD:
      return scored[0][1].title
    return "unassigned"

  def _on_region_select(self, row):
    if row < 0 or not self._regions or row >= len(self._regions):
      self._active_region = None
      self._picker.set_region_label(None)
      self._refresh_region_actions()
      return
    self._store_current_choice()
    self._active_region = row
    self._picker.set_region_label(row)
    region = self._regions[row]
    self._refresh_picker_context(region)
    self._fill_query_fields(row, region)

    if row in self._region_scored:
      self._picker.populate(self._region_scored[row])
    elif row in self._region_chosen:
      self._picker.populate([self._region_chosen[row]])  # restored, no re-search
    else:
      self._picker.populate([])
      self._picker.search()

    # register the auto-selected top candidate (enables Align region); the
    # emit is suppressed during populate so cues on screen are untouched
    self._store_current_choice()

    self._timeline.canvas.zoom_to_span(region.start, region.end)
    self._refresh_region_actions()

  def _refresh_picker_context(self, region):
    """Point the picker's scoring at this region's transcript slice."""
    from ..core.regions import region_segments
    self._picker.set_context(region_segments(self._segments, region),
                             int(region.duration * 1000))

  def _fill_query_fields(self, row, region):
    """Seed the picker's Title/Album/Artist from this region's metadata: its
    assigned pick first, then a confident detection, else (multi-song files
    only) an ASR-mined query. Single-region files keep the container tags."""
    chosen = self._region_chosen.get(row)
    cand = chosen[1] if chosen else None
    if cand is None:
      scored = self._region_scored.get(row) or []
      if scored and scored[0][0] >= SIMILARITY_THRESHOLD:
        cand = scored[0][1]
    if cand is not None:
      for combo, val in ((self._picker.title_combo, cand.title),
                         (self._picker.album_combo, getattr(cand, "album", "")),
                         (self._picker.artist_combo, getattr(cand, "artist", ""))):
        if val:
          combo.lineEdit().setText(val)
      return
    if len(self._regions) > 1:
      from ..core.regions import region_query
      self._picker.title_combo.lineEdit().setText(
        region_query(self._segments, region))

  # --- region add / delete / split ---

  def _on_region_context_menu(self, pos):
    if not self._media_path:
      return
    item = self._region_list.itemAt(pos)
    row = self._region_list.indexOfTopLevelItem(item) if item else -1
    playhead = self._timeline.canvas.playhead
    menu = QMenu(self._region_list)
    add_act = menu.addAction("Add region at playhead")
    split_act = menu.addAction("Split region at playhead")
    del_act = menu.addAction("Delete region")
    bounds_act = menu.addAction("Edit bounds...")
    bounds_act.setEnabled(row >= 0)
    split_act.setEnabled(row >= 0)
    del_act.setEnabled(row >= 0)
    chosen = menu.exec(self._region_list.viewport().mapToGlobal(pos))
    if chosen is add_act:
      self._add_region(playhead)
    elif chosen is split_act:
      self._split_region(row, playhead)
    elif chosen is del_act:
      self._delete_region(row)
    elif chosen is bounds_act:
      self._edit_region_bounds(row)

  def region_op(self, op):
    """Menu entries: add/split/delete on the active region. The dock's own
    context menu targets the clicked row instead."""
    if not self._media_path:
      return
    playhead = self._timeline.canvas.playhead
    if op == "add":
      self._add_region(playhead)
    elif self._active_region is None:
      return
    elif op == "split":
      self._split_region(self._active_region, playhead)
    else:
      self._delete_region(self._active_region)

  def _region_index_at(self, t):
    """Index of the region containing t, or -1."""
    for i, r in enumerate(self._regions):
      if r.start <= t <= r.end:
        return i
    return -1

  def _commit_regions(self, select_row=None):
    """Push region-list edits to canvas, list, session; keep maps aligned."""
    self._timeline.canvas.regions = list(self._regions)
    self._sync_timeline_session()
    if select_row is not None:
      self._active_region = min(max(select_row, 0), len(self._regions) - 1) \
        if self._regions else None
    self._populate_region_list()
    if self._active_region is not None:
      self._select_region_row(self._active_region)
    self._timeline.canvas.update()
    self._refresh_region_actions()
    self._timeline.canvas._set_dirty(True)

  def _shift_region_maps(self, at, delta):
    """Renumber per-region dicts/metas after an insert (delta=+1) or delete."""
    def moved(idx):
      return idx + delta if idx >= at else idx
    self._region_scored = {moved(i): v for i, v in self._region_scored.items()
                           if delta > 0 or i != at}
    self._region_chosen = {moved(i): v for i, v in self._region_chosen.items()
                           if delta > 0 or i != at}
    metas = []
    for m in self._region_metas:
      i = m.get("region_idx", -1)
      if delta < 0 and i == at:
        continue
      m = dict(m, region_idx=moved(i))
      metas.append(m)
    self._region_metas = metas

  def _add_region(self, t):
    """New region starting at the playhead, ending at the next region start
    (or the media end), skipped when it would overlap an existing region."""
    from ..core.regions import Region
    from .timeline_util import MIN_REGION_S
    if self._region_index_at(t) >= 0:
      self._say("  add region: playhead is inside an existing region")
      return
    end = self._timeline.canvas.duration or (t + 60.0)
    for r in self._regions:
      if r.start > t:
        end = min(end, r.start)
        break
    if end - t < MIN_REGION_S:
      self._say("  add region: not enough room after the playhead")
      return
    at = sum(1 for r in self._regions if r.start < t)
    self._regions.insert(at, Region(round(t, 3), round(end, 3)))
    self._shift_region_maps(at, +1)
    print(f"  added region {at + 1}")
    self._commit_regions(select_row=at)

  def _split_region(self, row, t):
    """Split region `row` at the playhead."""
    from ..core.regions import Region
    from .timeline_util import MIN_REGION_S
    if row < 0 or row >= len(self._regions):
      return
    r = self._regions[row]
    if not (r.start + MIN_REGION_S <= t <= r.end - MIN_REGION_S):
      self._say("  split region: playhead too close to a boundary")
      return
    tail = Region(round(t, 3), r.end, r.candidate_idx)
    r.end = round(t, 3)
    self._regions.insert(row + 1, tail)
    self._shift_region_maps(row + 1, +1)
    print(f"  split region {row + 1}")
    self._commit_regions(select_row=row)

  def _delete_region(self, row):
    """Drop region `row` and its assignment/meta."""
    if row < 0 or row >= len(self._regions):
      return
    if row in self._region_chosen:
      reply = QMessageBox.question(
        self, "Delete region",
        f"Region {row + 1} has lyrics assigned ({self._region_title(row)}).\nDelete it?",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
      if reply != QMessageBox.Yes:
        return
    self._regions.pop(row)
    self._shift_region_maps(row, -1)
    print(f"  deleted region {row + 1}")
    if not self._regions:
      self._active_region = None
      self._region_list.clear()
    self._commit_regions(select_row=row if self._regions else None)
    self._rebuild_provisional_cues()

  # --- bounds ---

  def _edit_region_bounds(self, idx):
    if idx is None or not 0 <= idx < len(self._regions):
      return
    from .timeline_util import parse_time, MIN_REGION_S
    region = self._regions[idx]
    dialog = QDialog(self)
    dialog.setWindowTitle(f"Region {idx + 1}: Edit bounds")
    form = QFormLayout(dialog)
    start_edit = QLineEdit(f"{int(region.start // 60)}:{region.start % 60:06.3f}")
    end_edit = QLineEdit(f"{int(region.end // 60)}:{region.end % 60:06.3f}")
    start_text, end_text = start_edit.text(), end_edit.text()
    form.addRow("Start", start_edit)
    form.addRow("End", end_edit)
    error = QLabel()
    error.setWordWrap(True)
    form.addRow(error)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    form.addRow(buttons)

    def apply_bounds():
      start, end = parse_time(start_edit.text()), parse_time(end_edit.text())
      if start_edit.text() == start_text:
        start = region.start
      if end_edit.text() == end_text:
        end = region.end
      lower = self._regions[idx - 1].end if idx else 0.0
      upper = (self._regions[idx + 1].start if idx + 1 < len(self._regions)
               else self._timeline.canvas.duration or float("inf"))
      if start is None or end is None or not lower <= start < end <= upper or end - start < MIN_REGION_S:
        error.setText("Enter valid times within the media and neighboring regions.")
        return
      if (start, end) != (region.start, region.end):
        region.start, region.end = round(start, 3), round(end, 3)
        self._on_region_bounds_updated(idx)
      dialog.accept()

    buttons.accepted.connect(apply_bounds)
    buttons.rejected.connect(dialog.reject)
    dialog.exec()

  def _on_region_boundary_dragged(self, ridx):
    if ridx < 0 or ridx >= len(self._regions):
      return
    self._on_region_bounds_updated(ridx)

  def _on_region_bounds_updated(self, idx):
    region = self._regions[idx]
    if idx == self._active_region:
      self._refresh_picker_context(region)
    self._populate_region_list()
    self._timeline.canvas.regions = list(self._regions)
    self._timeline.canvas.update()
    self._refresh_timeline_session_meta()
    self._timeline.canvas._set_dirty(True)
