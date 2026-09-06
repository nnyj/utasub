"""Cue grid: Aegisub-style editable table below the waveform."""
import math
from dataclasses import replace

from PySide6.QtCore import Qt, Signal, QItemSelectionModel
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (
  QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QMenu,
)

from . import theme
from .timeline_util import romaji_for, fmt_time_mssff, parse_time, MIN_CUE_LEN_S

CONF_HINT = (
  "Conf: how sure the CTC aligner was of this line's own stamp.\n"
  "Blank means no stamp survived and the line was placed by warp + onset snap.\n"
  "Amber rows are this song's bottom 10%, relative because a speech aligner\n"
  "scores all singing low: Alt+N / Alt+P jump between them.")

class CueGrid(QTableWidget):
  """Editable cue list: # | start | end | conf | text | romaji."""

  cue_edited = Signal(int, object)   # (idx, new_cue) after inline edit
  selection_sync = Signal(int)       # row selected by user click
  editing_changed = Signal(bool)     # inline cell editor opened / closed
  recalc_requested = Signal(list)    # row indices to re-stamp via CTC

  COL_NUM = 0
  COL_START = 1
  COL_END = 2
  COL_CONF = 3
  COL_TEXT = 4
  COL_ROMAJI = 5

  def __init__(self, parent=None):
    super().__init__(0, 6, parent)
    self.setHorizontalHeaderLabels(["#", "Start", "End", "Conf", "Text", "Romaji"])
    self.horizontalHeader().setToolTip(CONF_HINT)
    self.setSelectionBehavior(QAbstractItemView.SelectRows)
    self.setSelectionMode(QAbstractItemView.ExtendedSelection)
    # Text stays user-resizable; Romaji absorbs the leftover width
    self.horizontalHeader().setSectionResizeMode(self.COL_TEXT, QHeaderView.Interactive)
    self.horizontalHeader().setSectionResizeMode(self.COL_ROMAJI, QHeaderView.Stretch)
    self.verticalHeader().setVisible(False)
    self.setColumnWidth(self.COL_NUM, 40)
    self.setColumnWidth(self.COL_START, 80)
    self.setColumnWidth(self.COL_END, 80)
    self.setColumnWidth(self.COL_CONF, 52)
    self.setColumnWidth(self.COL_TEXT, 320)

    self._cues = []  # reference to canvas.cues
    self._conf_cut = None  # per-song low-confidence percentile, set on rebuild
    self._provisional = False  # dim + read-only-ish display
    self._syncing = False  # guard against recursive updates
    self._playing_row = -1  # row tinted to match the canvas playback glow
    self.follow_playhead = True  # auto-scroll to the playing row (Follow toggle)
    self.cellChanged.connect(self._on_cell_changed)
    self.currentCellChanged.connect(self._on_current_changed)

  # An open cell editor has to own the keys the timeline panel also binds (N, P,
  # arrows), so it announces itself and actions.py stands the shortcuts down.

  def edit(self, index, trigger=QAbstractItemView.AllEditTriggers, event=None):
    started = super().edit(index, trigger, event)
    if started:
      self.editing_changed.emit(True)
    return started

  def closeEditor(self, editor, hint):
    super().closeEditor(editor, hint)
    self.editing_changed.emit(False)

  def load_cues(self, cues):
    """Populate grid from cue list."""
    self._cues = cues
    self.rebuild()

  def set_provisional(self, flag):
    """Dim rows + ignore edits while showing provisional cues."""
    self._provisional = flag
    self.rebuild()

  def rebuild(self):
    """Full rebuild from self._cues. Call after structural changes (insert/remove)."""
    from ..core.ctc_align import low_conf_cut
    self._conf_cut = low_conf_cut([getattr(c, "score", None) for c in self._cues])
    self._syncing = True
    self.setRowCount(len(self._cues))
    for i, c in enumerate(self._cues):
      self._set_row(i, c)
    self._syncing = False
    playing, self._playing_row = self._playing_row, -1  # rebuilt items carry no tint
    self.set_playing_row(playing)

  def _set_row(self, row, cue):
    """Set cells for one row. Romaji auto-derived."""
    num_item = QTableWidgetItem(str(row + 1))
    num_item.setFlags(num_item.flags() & ~Qt.ItemIsEditable)
    self.setItem(row, self.COL_NUM, num_item)
    self.setItem(row, self.COL_START, QTableWidgetItem(fmt_time_mssff(cue.start)))
    self.setItem(row, self.COL_END, QTableWidgetItem(fmt_time_mssff(cue.end)))
    self.setItem(row, self.COL_CONF, self._conf_item(cue))
    self.setItem(row, self.COL_TEXT, QTableWidgetItem(cue.text))
    rom_item = QTableWidgetItem(romaji_for(cue.text))
    rom_item.setFlags(rom_item.flags() & ~Qt.ItemIsEditable)
    rom_item.setForeground(QBrush(theme.GRAY))
    self.setItem(row, self.COL_ROMAJI, rom_item)

    # provisional: gray + italic across the row
    if self._provisional:
      dim_font = self.font()
      dim_font.setItalic(True)
      for col in range(self.columnCount()):
        it = self.item(row, col)
        if it:
          it.setForeground(QBrush(theme.CREDIT_STRIKE))
          it.setFont(dim_font)

  def _conf_brush(self, row):
    """Conf-cell background for a row, amber below this song's cut."""
    score = getattr(self._cues[row], "score", None) if row < len(self._cues) else None
    if score is None:
      return QBrush(theme.CONF_NONE_BG)
    low = self._conf_cut is not None and score < self._conf_cut
    return QBrush(theme.CONF_LOW_BG) if low else QBrush(Qt.NoBrush)

  def _conf_item(self, cue):
    """Conf cell: the CTC score, tinted by how far down this song's own ranking
    it sits. Blank and untinted when the line has no stamp, which is not the
    same as a bad one. The stored score is a mean token log-prob, so it is
    exponentiated back into a 0..100% reading before it is shown."""
    score = getattr(cue, "score", None)
    item = QTableWidgetItem(
      "" if score is None else f"{round(100 * min(1.0, math.exp(score)))}%")
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if score is None:
      item.setForeground(QBrush(theme.DIM))
      item.setBackground(QBrush(theme.CONF_NONE_BG))
      item.setToolTip("no stamp: Recalc syllables to fill karaoke timing")
      return item
    low = self._conf_cut is not None and score < self._conf_cut
    item.setForeground(QBrush(theme.CONF_LOW if low else theme.GRAY))
    if low:
      item.setBackground(QBrush(theme.CONF_LOW_BG))
      item.setToolTip("low: this song's bottom 10%, worth a listen")
    return item

  def update_row(self, idx):
    """Update single row after canvas edit (no structural change)."""
    if idx < 0 or idx >= len(self._cues):
      return
    self._syncing = True
    self._set_row(idx, self._cues[idx])
    self._syncing = False

  def set_playing_row(self, idx):
    """Tint row under playhead (mirrors canvas glow), scroll into view.
    idx < 0 clears the tint."""
    if idx == self._playing_row:
      return
    self._syncing = True  # background writes emit cellChanged otherwise
    for row, brush in ((self._playing_row, QBrush(Qt.NoBrush)),
                       (idx, QBrush(theme.PLAYING_ROW))):
      if 0 <= row < self.rowCount():
        for col in range(self.columnCount()):
          it = self.item(row, col)
          if it:
            # the Conf tint is per-cue, so it has to survive the playhead
            # sweeping the row rather than be cleared with it
            it.setBackground(self._conf_brush(row) if
                             (col == self.COL_CONF and brush.style() == Qt.NoBrush)
                             else brush)
    self._syncing = False
    self._playing_row = idx
    if self.follow_playhead and 0 <= idx < self.rowCount():
      self.scrollToItem(self.item(idx, 0), QAbstractItemView.EnsureVisible)

  def select_row(self, idx):
    """Sync selection from canvas without emitting back."""
    if idx < 0 or idx >= self.rowCount():
      self.clearSelection()
      return
    self._syncing = True
    # a multi-selection that already holds the row is kept: the canvas echo of
    # a Shift/Ctrl click must not collapse it back to one row
    held = any(ix.row() == idx for ix in self.selectionModel().selectedRows())
    self.setCurrentCell(idx, 0, QItemSelectionModel.NoUpdate if held
                        else QItemSelectionModel.ClearAndSelect
                        | QItemSelectionModel.Rows)
    self._syncing = False

  def selected_rows(self):
    """Selected row indices, ascending; the current row when nothing is selected."""
    rows = sorted(ix.row() for ix in self.selectionModel().selectedRows())
    if not rows and self.currentRow() >= 0:
      rows = [self.currentRow()]
    return rows

  def mousePressEvent(self, ev):
    """Right-click on a row already in the selection keeps the selection, so
    the context menu acts on every row that was picked."""
    idx = self.indexAt(ev.pos())
    if ev.button() == Qt.RightButton and idx.isValid() and any(
        ix.row() == idx.row() for ix in self.selectionModel().selectedRows()):
      return
    super().mousePressEvent(ev)

  def contextMenuEvent(self, ev):
    """Right-click menu on the cue rows: re-stamp the selected cue(s)."""
    if self._provisional:
      return
    rows = self.selected_rows()
    if not rows:
      return
    menu = QMenu(self)
    act = menu.addAction("Recalc syllables")
    blank = [r for r in range(len(self._cues))
             if getattr(self._cues[r], "score", None) is None]
    act_blank = menu.addAction("Recalc syllables (blank only)")
    act_all = menu.addAction("Recalc syllables (all)")
    chosen = menu.exec(ev.globalPos())
    if chosen is act:
      self.recalc_requested.emit(rows)
    elif chosen is act_blank and blank:
      self.recalc_requested.emit(blank)
    elif chosen is act_all:
      self.recalc_requested.emit(list(range(len(self._cues))))

  def _on_current_changed(self, row, col, prev_row, prev_col):
    """User clicked a different row: emit for canvas sync."""
    if self._syncing or row < 0:
      return
    self.selection_sync.emit(row)

  def _on_cell_changed(self, row, col):
    """User edited a cell: validate and emit cue_edited."""
    if self._syncing or self._provisional or row < 0 or row >= len(self._cues):
      return
    c = self._cues[row]
    item = self.item(row, col)
    if item is None:
      return

    if col in (self.COL_START, self.COL_END):
      start = col == self.COL_START
      t = parse_time(item.text())
      ok = t is not None and (t < c.end - MIN_CUE_LEN_S if start
                              else t > c.start + MIN_CUE_LEN_S)
      if ok:
        self.cue_edited.emit(row, replace(c, **{"start" if start else "end": t}))
        return
      self._syncing = True
      item.setText(fmt_time_mssff(c.start if start else c.end))
      self._syncing = False

    elif col == self.COL_TEXT:
      new_text = item.text()
      if new_text != c.text:
        new_cue = replace(c, text=new_text)
        self.cue_edited.emit(row, new_cue)
        self._syncing = True
        rom_item = self.item(row, self.COL_ROMAJI)
        if rom_item:
          rom_item.setText(romaji_for(new_text))
        self._syncing = False
