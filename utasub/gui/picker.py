"""Lyric picker panel: combo fields with prefill alternates, auto-search,
ranked candidate table with similarity %, preview pane, paste-row.
Embedded per-region in MainWindow; region master-detail lives there."""
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QBrush, QKeySequence, QShortcut, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
  QHBoxLayout, QVBoxLayout, QFormLayout,
  QComboBox, QPushButton, QLabel,
  QTableWidget, QTableWidgetItem, QHeaderView,
  QTextEdit, QSplitter,
  QAbstractItemView, QWidget, QListWidget, QListWidgetItem,
)

from ..core.align import SIMILARITY_THRESHOLD
from . import theme
from .theme import GRAY
# build_alternates lives in core.providers: core.multi_song needs it headless
from ..core.providers import build_alternates  # noqa: F401

# --- search worker thread ---

class SearchWorker(QThread):
  """Fetch candidates in background. Emits results or error."""
  finished = Signal(list)  # list of (score, Candidate)
  error = Signal(str)
  status = Signal(str)

  def __init__(self, query, segments, media_duration_ms,
               providers=None, fresh=False, region_idx=None):
    super().__init__()
    self.region_idx = region_idx  # region the search was started for
    self.query = query
    self.segments = segments
    self.media_duration_ms = media_duration_ms
    self.providers = providers or ["NetEase"]
    self.fresh = fresh

  def run(self):
    try:
      from ..core.providers import fetch_from_providers
      from ..core.align import score_candidates

      self.status.emit("Searching...")
      candidates = fetch_from_providers(self.query, self.providers, self.fresh)

      scored = score_candidates(candidates, self.segments, self.media_duration_ms)
      self.finished.emit(scored)
    except Exception as e:
      self.error.emit(str(e))

class _PasteEdit(QTextEdit):
  """Preview/paste pane: pastes land as plain text, so no strikethrough or
  colors are inherited from the source."""

  def __init__(self, parent=None):
    super().__init__(parent)
    self.setAcceptRichText(False)

  def insertFromMimeData(self, source):
    self.setCurrentCharFormat(QTextCharFormat())  # drop any leftover strike/color
    self.insertPlainText(source.text())

# --- picker panel (reusable widget) ---

class PickerPanel(QWidget):
  """Search fields, candidate table, LRC preview, credit toggles, paste-row.
  Embeddable in the MainWindow picker tab."""

  results_ready = Signal(list, object)  # [(score, Candidate), ...], region idx
  apply_requested = Signal()     # "Apply to region" pressed

  EMPTY_HINT = "No candidates. Check the title and press Search."

  def __init__(self, parent=None, *,
               media_path=None, tags=None, segments=None,
               media_duration_ms=0, providers=None, fresh=False):
    super().__init__(parent)
    self.media_path = media_path
    self.segments = segments or []
    self.media_duration_ms = media_duration_ms
    self.providers = providers or ["NetEase"]
    self.fresh = fresh
    self._worker = None
    self._scored = []  # [(score, Candidate), ...]
    self._chosen_index = None  # index into _scored, or -1 for paste
    self._paste_text = ""
    self._verdict_cache = {}  # row_index -> (lines, verdicts)
    self._stale_workers = []  # superseded SearchWorkers still running
    self._region_idx = None   # region a search/apply targets

    alts = build_alternates(media_path or "", tags)

    # --- top row: query fields ---
    top = QHBoxLayout()
    form = QFormLayout()
    self.title_combo = self._make_combo(alts["title"])
    self.album_combo = self._make_combo(alts["album"])
    self.artist_combo = self._make_combo(alts["artist"])
    form.addRow("Title:", self.title_combo)
    form.addRow("Album:", self.album_combo)
    form.addRow("Artist:", self.artist_combo)
    for combo in (self.title_combo, self.album_combo, self.artist_combo):
      combo.lineEdit().returnPressed.connect(self.search)  # Enter = Search
    top.addLayout(form, 1)

    btn_col = QVBoxLayout()
    self.search_btn = QPushButton("Search")
    self.search_btn.setToolTip("Search providers (Enter in any field)")
    self.search_btn.clicked.connect(self.search)
    self.status_label = QLabel(self.EMPTY_HINT)
    self.status_label.setWordWrap(True)
    self.status_label.setStyleSheet(f"color: {theme.HINT_GRAY};")
    btn_col.addWidget(self.search_btn)
    btn_col.addWidget(self.status_label)
    btn_col.addStretch()
    top.addLayout(btn_col)

    # --- center: table + preview splitter ---
    self.table = QTableWidget(0, 6)
    self.table.setHorizontalHeaderLabels(
      ["Title", "Album", "Artist", "Source", "Sim %", "Duration"])
    # Title absorbs the spare width; the rest sit at content width, user-draggable
    hh = self.table.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.Interactive)
    hh.setSectionResizeMode(0, QHeaderView.Stretch)
    for col in (3, 4, 5):
      hh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
    self.table.setColumnWidth(1, 160)  # Album, Artist: draggable defaults
    self.table.setColumnWidth(2, 140)
    self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
    self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
    self.table.setSelectionMode(QAbstractItemView.SingleSelection)
    self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    self.table.verticalHeader().setVisible(False)
    self.table.selectionModel().selectionChanged.connect(self._on_select)

    self.preview = _PasteEdit()
    self.preview.setReadOnly(True)
    self.preview.setLineWrapMode(QTextEdit.NoWrap)
    self.preview.textChanged.connect(self._refresh_apply_btn)  # paste-row typing

    # credit toggle list (shown below preview when verdicts exist)
    self._credit_verdicts = []  # parallel to parsed lines
    self._credit_lines = []  # [(time_or_None, text), ...]
    self.credit_list = QListWidget()
    self.credit_list.itemChanged.connect(self._on_credit_toggle)
    self.credit_list.setVisible(False)

    # the only path that assigns lyrics: selecting a row previews it, this applies it
    self.apply_btn = QPushButton("Apply to region")
    self.apply_btn.setToolTip(
      "Assign the selected lyrics to the active region and replace the timeline cues")
    self.apply_btn.setEnabled(False)
    self.apply_btn.clicked.connect(self._on_apply)
    QShortcut(QKeySequence("Ctrl+Return"), self,
              activated=self._on_apply,
              context=Qt.WidgetWithChildrenShortcut)

    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.addWidget(self.preview, 3)
    right_layout.addWidget(self.apply_btn)
    self._credit_label = QLabel(
      "struck lines = detected credits, excluded from subs, toggle:")
    self._credit_label.setStyleSheet(f"color: {theme.HINT_GRAY};")
    self._credit_label.setVisible(False)
    right_layout.addWidget(self._credit_label)
    right_layout.addWidget(self.credit_list, 2)

    splitter = QSplitter(Qt.Horizontal)
    splitter.addWidget(self.table)
    splitter.addWidget(right_panel)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 2)

    # --- layout ---
    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addLayout(top)
    layout.addWidget(splitter, 1)

  # --- context update (for region switching) ---

  def set_context(self, segments, media_duration_ms):
    """Update ASR segments and duration for a different region."""
    self.segments = segments
    self.media_duration_ms = media_duration_ms

  def prefill(self, media_path, tags=None):
    """Reseed Title/Album/Artist from container tags + filename alternates."""
    alts = build_alternates(str(media_path), tags)
    for combo, key in ((self.title_combo, "title"), (self.album_combo, "album"),
                       (self.artist_combo, "artist")):
      combo.clear()
      self._seed_combo(combo, alts[key])

  # --- combo helper ---

  @staticmethod
  def _seed_combo(combo, alternates):
    """Fill an editable combo from [(value, source_hint), ...]."""
    for val, hint in alternates:
      combo.addItem(f"{val}  ({hint})" if hint else val, val)
    if alternates:
      combo.setCurrentIndex(0)
      combo.lineEdit().setText(alternates[0][0])

  @classmethod
  def _make_combo(cls, alternates):
    """Editable combo from [(value, source_hint), ...]."""
    combo = QComboBox()
    combo.setEditable(True)
    # picking an item shows the hint-decorated label; swap in the raw value so
    # the field reads clean (the query already strips the hint)
    combo.activated.connect(
      lambda idx, c=combo: c.lineEdit().setText(c.itemData(idx) or c.itemText(idx)))
    cls._seed_combo(combo, alternates)
    combo.setMinimumWidth(140)
    return combo

  def _combo_value(self, combo):
    """Current raw text (strip hint if user picked from dropdown)."""
    idx = combo.currentIndex()
    data = combo.itemData(idx)
    if data is not None and combo.currentText() == combo.itemText(idx):
      return data
    return combo.currentText().strip()

  # --- worker lifecycle ---

  def stop(self):
    """Detach the current SearchWorker. Its blocking network fetch cannot be
    interrupted, so a running thread is parked (referenced until it finishes,
    else Qt destroys a live QThread and crashes) and its results are ignored
    via the sender check in _on_results/_on_error."""
    w = self._worker
    self._worker = None
    if w is not None and w.isRunning():
      self._stale_workers.append(w)
    self._stale_workers = [x for x in self._stale_workers if x.isRunning()]

  def shutdown(self):
    """Blocking wait for every search thread; call only at window close."""
    self.stop()
    for w in self._stale_workers:
      w.wait(3000)
    self._stale_workers = []

  # --- search ---

  def search(self):
    title = self._combo_value(self.title_combo)
    artist = self._combo_value(self.artist_combo)
    query = f"{title} {artist}".strip()
    if not query:
      self.status_label.setText("Enter title or artist")
      return
    self.stop()
    self.search_btn.setEnabled(False)
    self.status_label.setText("Searching...")
    self._worker = SearchWorker(
      query, self.segments, self.media_duration_ms,
      self.providers, self.fresh, region_idx=self._region_idx)
    self._worker.finished.connect(self._on_results)
    self._worker.error.connect(self._on_error)
    self._worker.status.connect(self._on_status)
    self._worker.start()

  def _on_results(self, scored):
    if self.sender() is not self._worker:
      return  # parked worker from a superseded search
    region_idx = self._worker.region_idx
    self.search_btn.setEnabled(True)
    self.populate(scored)
    if scored:
      above = sum(1 for s, _ in scored if s >= SIMILARITY_THRESHOLD)
      self.status_label.setText(f"{len(scored)} results ({above} above threshold)")
    # the region may have changed while the fetch ran; the host drops mismatches
    self.results_ready.emit(scored, region_idx)

  def _on_status(self, msg):
    if self.sender() is self._worker:
      self.status_label.setText(msg)

  def _on_error(self, msg):
    if self.sender() is not self._worker:
      return
    self.search_btn.setEnabled(True)
    self.status_label.setText(f"Error: {msg}")

  # --- table population ---

  def populate(self, scored):
    """Fill table from [(score, Candidate), ...]. Appends paste-row."""
    from ..core.romanize import romanize_suffix

    self._scored = scored
    self._verdict_cache = {}
    self.table.setRowCount(len(scored) + 1)
    for row, (score, c) in enumerate(scored):
      items = [
        c.title, c.album, c.artist, c.source,
        f"{score * 100:.0f}%",
        f"{c.duration_ms // 60000}:{c.duration_ms // 1000 % 60:02d}" if c.duration_ms else "",
      ]
      gray = score < SIMILARITY_THRESHOLD
      for col, text in enumerate(items):
        # romaji suffix in title/album/artist cells
        suffix = romanize_suffix(text) if col < 3 else ""
        item = QTableWidgetItem(text + suffix)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        if suffix:
          item.setToolTip(text)  # original in tooltip for copy
        if gray:
          item.setForeground(QBrush(GRAY))
        self.table.setItem(row, col, item)

    paste_row = len(scored)
    paste_items = ["(paste lyrics)", "", "", "Manual", "", ""]
    for col, text in enumerate(paste_items):
      item = QTableWidgetItem(text)
      item.setFlags(item.flags() & ~Qt.ItemIsEditable)
      item.setForeground(QBrush(theme.PASTE_ROW))
      self.table.setItem(paste_row, col, item)

    if scored:
      self.table.selectRow(0)
    else:
      self.status_label.setText(self.EMPTY_HINT)
    self._refresh_apply_btn()

  # --- selection / preview ---

  def _on_select(self, selected, deselected):
    # leaving the paste-row: stash edits so they survive row/region switches
    if self._chosen_index == -1:
      self._paste_text = self.preview.toPlainText()
    rows = self.table.selectionModel().selectedRows()
    if not rows:
      self._chosen_index = None
      self.preview.setPlainText("")
      self._hide_credit_panel()
      self._refresh_apply_btn()
      return
    row = rows[0].row()
    paste_row = len(self._scored)
    if row == paste_row:
      # paste-row: enable editing on a clean (unformatted) document
      self._chosen_index = -1
      self.preview.setReadOnly(False)
      self.preview.clear()
      self.preview.setCurrentCharFormat(QTextCharFormat())
      self.preview.setPlainText(self._paste_text)
      self.preview.setPlaceholderText("Paste LRC lyrics here...")
      self._hide_credit_panel()
    elif 0 <= row < len(self._scored):
      self._chosen_index = row
      self.preview.setReadOnly(True)
      self._show_with_credits(self._scored[row][1].lrc, row_index=row)
    else:
      self._chosen_index = None
      self.preview.setPlainText("")
      self.preview.setReadOnly(True)
      self._hide_credit_panel()
    self._refresh_apply_btn()

  # --- apply ---

  def set_region_label(self, region_idx):
    """Name the target region on the apply button; None = no active region."""
    self._region_idx = region_idx
    suffix = "" if region_idx is None else f" {region_idx + 1}"
    self.apply_btn.setText(f"Apply to region{suffix}  (Ctrl+Enter)")

  def is_paste_row(self):
    return self._chosen_index == -1

  def _refresh_apply_btn(self):
    """Enabled while a candidate row is selected, or the paste-row has text."""
    self.apply_btn.setEnabled(self.chosen_candidate() is not None)

  def _on_apply(self):
    """Assign the selected row to the active region (host does the work)."""
    if not self.apply_btn.isEnabled():
      return
    if self.is_paste_row():
      self._paste_text = self.preview.toPlainText()
    self.apply_requested.emit()

  # --- credit display ---

  def _render_preview(self):
    """Render LRC preview with strikethrough on credit lines, romaji interleaved."""
    from ..core.romanize import is_cjk, romanize

    self.preview.clear()
    cursor = self.preview.textCursor()
    normal_fmt = QTextCharFormat()
    strike_fmt = QTextCharFormat()
    strike_fmt.setFontStrikeOut(True)
    strike_fmt.setForeground(QBrush(theme.CREDIT_STRIKE))
    romaji_fmt = QTextCharFormat()
    romaji_fmt.setForeground(QBrush(theme.ROMAJI_CYAN))  # cyan for romaji
    for i, ((t, text), v) in enumerate(zip(self._credit_lines, self._credit_verdicts)):
      if i > 0:
        cursor.insertText("\n")
      prefix = f"[{int(t // 60):02d}:{t % 60:05.2f}] " if t is not None else ""
      fmt = strike_fmt if v == "credit" else normal_fmt
      cursor.insertText(prefix + text, fmt)
      # romaji interleave for Japanese lines (not struck credits)
      if v != "credit" and is_cjk(text):
        rom = romanize(text)
        if rom and rom != text:
          cursor.insertText("\n")
          cursor.insertText(prefix + rom, romaji_fmt)
    self.preview.setTextCursor(cursor)
    self.preview.moveCursor(QTextCursor.Start)

  def _show_with_credits(self, lrc_text, row_index=None):
    """Show LRC in preview with strikethrough on credit lines, populate toggle list.
    row_index: cache key for classify verdicts (avoids re-running on selection change)."""
    from lyrickit import classify_lines
    from ..core.providers import parse_lrc

    lines = parse_lrc(lrc_text)
    if not lines:
      self.preview.setPlainText(lrc_text)
      self._hide_credit_panel()
      return

    # cached verdicts survive row switches; a length change means a different
    # parse, so re-classify
    cached = self._verdict_cache.get(row_index)
    if cached and len(cached[0]) == len(lines):
      lines, verdicts = cached
    else:
      verdicts = classify_lines(lines)
      if row_index is not None:
        self._verdict_cache[row_index] = (lines, verdicts)

    self._credit_verdicts = verdicts
    self._credit_lines = lines
    # pin locale from the whole candidate lyric before per-line preview romaji
    from ..core.romanize import detect, set_default_locale
    set_default_locale(detect("\n".join(t for _, t in lines)))

    self._render_preview()

    has_credits = any(v == "credit" for v in verdicts)
    self._credit_label.setVisible(has_credits)
    self.credit_list.setVisible(has_credits)
    if not has_credits:
      return

    self.credit_list.blockSignals(True)
    self.credit_list.clear()
    for i, ((t, text), v) in enumerate(zip(lines, verdicts)):
      if v == "credit":
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)  # checked = struck
        item.setData(Qt.UserRole, i)
        self.credit_list.addItem(item)
    self.credit_list.blockSignals(False)

  def _hide_credit_panel(self):
    self.credit_list.setVisible(False)
    self._credit_label.setVisible(False)
    self._credit_verdicts = []
    self._credit_lines = []

  def _on_credit_toggle(self, item):
    """User toggled a credit line: update verdict + refresh preview."""
    idx = item.data(Qt.UserRole)
    if idx is None or idx >= len(self._credit_verdicts):
      return
    self._credit_verdicts[idx] = "credit" if item.checkState() == Qt.Checked else "sung"
    self._render_preview()

  # --- result access ---

  def chosen_candidate(self):
    """Returns (score, Candidate) or None. For paste-row, returns
    a synthetic Candidate with user-pasted LRC."""
    if self._chosen_index is None:
      return None
    if self._chosen_index == -1:
      # paste-row
      from ..core.providers import Candidate
      text = self.preview.toPlainText().strip()
      if not text:
        return None
      return (0.0, Candidate(
        title=self._combo_value(self.title_combo),
        album=self._combo_value(self.album_combo),
        artist=self._combo_value(self.artist_combo),
        source="Manual",
        lrc=text,
      ))
    if 0 <= self._chosen_index < len(self._scored):
      return self._scored[self._chosen_index]
    return None

