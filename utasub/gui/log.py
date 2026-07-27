"""Stdout tee + Qt log dock: print() reaches terminal AND GUI.
Safe when sys.stdout is None (pythonw): dock becomes sole output."""
import re
import sys

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QBrush, QFont, QTextBlockFormat, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QDockWidget, QPlainTextEdit

from . import theme

_original_stdout = None


class LogBridge(QObject):
  """Thread-safe bridge: emits line(str) on queued connection."""
  line = Signal(str)


class TeeStream:
  """Wraps sys.stdout so writes go to real terminal + LogBridge signal.
  If real_stdout is None (pythonw), only emits to the bridge."""

  def __init__(self, real_stdout, bridge):
    self._real = real_stdout
    self._bridge = bridge
    self._buf = ""

  def write(self, text):
    if not text:
      return
    if self._real is not None:
      try:
        self._real.write(text)
      except (OSError, ValueError):
        pass  # handle closed/invalid stdout
    # buffer partials: print() writes text and newline separately
    self._buf += text
    while "\n" in self._buf:
      line, self._buf = self._buf.split("\n", 1)
      self._bridge.line.emit(line.rstrip("\r"))

  def flush(self):
    if self._buf:
      self._bridge.line.emit(self._buf.rstrip("\r"))
      self._buf = ""
    if self._real is not None:
      try:
        self._real.flush()
      except (OSError, ValueError):
        pass

  def fileno(self):
    if self._real is not None and hasattr(self._real, 'fileno'):
      return self._real.fileno()
    raise OSError("no underlying file descriptor")

  @property
  def encoding(self):
    return getattr(self._real, "encoding", "utf-8")


def install_tee(bridge):
  """Replace sys.stdout with TeeStream. Safe to call once; handles
  pythonw where sys.stdout is None."""
  global _original_stdout
  if _original_stdout is not None:
    return  # already installed
  _original_stdout = sys.stdout  # may be None under pythonw
  sys.stdout = TeeStream(_original_stdout, bridge)
  # also guard stderr for print() fallback
  if sys.stderr is None:
    sys.stderr = TeeStream(None, bridge)


def uninstall_tee():
  """Restore original sys.stdout."""
  global _original_stdout
  if _original_stdout is not None or isinstance(sys.stdout, TeeStream):
    sys.stdout = _original_stdout
    _original_stdout = None


def _line_format(text):
  """Color a log line by its shape: headers, written files, problems, detail."""
  fmt = QTextCharFormat()
  stripped = text.strip()
  low = stripped.lower()
  if stripped.startswith("==="):
    fmt.setForeground(QBrush(theme.LOG_HEADER))
    fmt.setFontWeight(QFont.Bold)
  elif stripped.startswith("->"):
    fmt.setForeground(QBrush(theme.LOG_WRITE))
  elif ("error" in low or "failed" in low or "skip" in low
        or re.search(r"\bno\b", low)):  # word match: "no" not "piano"
    fmt.setForeground(QBrush(theme.LOG_WARN))
  elif text.startswith("  "):
    fmt.setForeground(QBrush(theme.LOG_DETAIL))
  return fmt


class LogDock(QDockWidget):
  """Read-only monospace log panel. Auto-scrolls only when at bottom."""

  def __init__(self, title="Log", parent=None):
    super().__init__(title, parent)
    self._edit = QPlainTextEdit()
    self._edit.setReadOnly(True)
    self._edit.setFont(QFont("Consolas", 9))
    self._edit.setMaximumBlockCount(5000)
    self._edit.setStyleSheet(
      f"QPlainTextEdit {{ background: {theme.LOG_BG}; color: {theme.LOG_FG}; }}")
    self.setWidget(self._edit)

    # tight line spacing: default blocks are loose for a log wall
    self._block_fmt = QTextBlockFormat()
    self._block_fmt.setLineHeight(88.0, QTextBlockFormat.ProportionalHeight.value)
    self._block_fmt.setTopMargin(0)
    self._block_fmt.setBottomMargin(0)
    cursor = self._edit.textCursor()
    cursor.setBlockFormat(self._block_fmt)  # first (empty) block

    self._bridge = LogBridge()
    self._bridge.line.connect(self._append, Qt.QueuedConnection)

  @property
  def bridge(self):
    return self._bridge

  def _append(self, text):
    sb = self._edit.verticalScrollBar()
    at_bottom = sb.value() >= sb.maximum() - 4
    doc = self._edit.document()
    cursor = QTextCursor(doc)
    cursor.movePosition(QTextCursor.End)
    if not doc.isEmpty():
      cursor.insertBlock(self._block_fmt)  # first line reuses the empty block
    cursor.insertText(text, _line_format(text))
    if at_bottom:
      sb.setValue(sb.maximum())
