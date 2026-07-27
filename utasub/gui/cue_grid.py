"""Cue grid: Aegisub-style editable table below the waveform."""
from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (
  QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from . import theme
from .timeline_util import romaji_for, fmt_time_mssff, parse_time


class CueGrid(QTableWidget):
  """Editable cue list: # | start | end | text | romaji."""

  cue_edited = Signal(int, object)   # (idx, new_cue) after inline edit
  selection_sync = Signal(int)       # row selected by user click

  COL_NUM = 0
  COL_START = 1
  COL_END = 2
  COL_TEXT = 3
  COL_ROMAJI = 4

  def __init__(self, parent=None):
    super().__init__(0, 5, parent)
    self.setHorizontalHeaderLabels(["#", "Start", "End", "Text", "Romaji"])
    self.setSelectionBehavior(QAbstractItemView.SelectRows)
    self.setSelectionMode(QAbstractItemView.SingleSelection)
    # Text stays user-resizable; Romaji absorbs the leftover width
    self.horizontalHeader().setSectionResizeMode(self.COL_TEXT, QHeaderView.Interactive)
    self.horizontalHeader().setSectionResizeMode(self.COL_ROMAJI, QHeaderView.Stretch)
    self.verticalHeader().setVisible(False)
    self.setColumnWidth(self.COL_NUM, 40)
    self.setColumnWidth(self.COL_START, 80)
    self.setColumnWidth(self.COL_END, 80)
    self.setColumnWidth(self.COL_TEXT, 320)

    self._cues = []  # reference to canvas.cues
    self._provisional = False  # dim + read-only-ish display
    self._syncing = False  # guard against recursive updates
    self._playing_row = -1  # row tinted to match the canvas playback glow
    self.follow_playhead = True  # auto-scroll to the playing row (Follow toggle)
    self.cellChanged.connect(self._on_cell_changed)
    self.currentCellChanged.connect(self._on_current_changed)

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
    self.setItem(row, self.COL_TEXT, QTableWidgetItem(cue.text))
    rom_item = QTableWidgetItem(romaji_for(cue.text))
    rom_item.setFlags(rom_item.flags() & ~Qt.ItemIsEditable)
    rom_item.setForeground(QBrush(theme.GRAY))
    self.setItem(row, self.COL_ROMAJI, rom_item)

    # provisional: gray + italic across the row
    if self._provisional:
      dim_font = self.font()
      dim_font.setItalic(True)
      for col in range(5):
        it = self.item(row, col)
        if it:
          it.setForeground(QBrush(theme.CREDIT_STRIKE))
          it.setFont(dim_font)

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
            it.setBackground(brush)
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
    self.setCurrentCell(idx, 0)
    self._syncing = False

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

    if col == self.COL_START:
      t = parse_time(item.text())
      if t is not None and t < c.end - 0.05:
        new_cue = replace(c, start=t)
        self.cue_edited.emit(row, new_cue)
      else:
        self._syncing = True
        item.setText(fmt_time_mssff(c.start))
        self._syncing = False

    elif col == self.COL_END:
      t = parse_time(item.text())
      if t is not None and t > c.start + 0.05:
        new_cue = replace(c, end=t)
        self.cue_edited.emit(row, new_cue)
      else:
        self._syncing = True
        item.setText(fmt_time_mssff(c.end))
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
