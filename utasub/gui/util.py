"""Small shared GUI helpers."""
from PySide6.QtCore import Qt, QObject, QEvent, QSettings
from PySide6.QtWidgets import QAbstractScrollArea

def settings():
  """User-level GUI prefs (registry on Windows, ini elsewhere). Per-file
  values live in session; this holds defaults for files without one."""
  return QSettings("utasub", "utasub")

def _shift_hscroll(view, ev):
  """Shift+wheel → horizontal scroll (Qt has no built-in mapping).
  view: any QAbstractScrollArea. Returns True when handled."""
  sb = view.horizontalScrollBar()
  if sb is None or sb.maximum() == 0:
    return False
  delta = ev.angleDelta().y() or ev.angleDelta().x()
  if not delta:
    return False
  sb.setValue(sb.value() - delta // 120 * max(sb.singleStep(), 1) * 3)
  ev.accept()
  return True

class _ShiftHScrollFilter(QObject):
  """App-wide shift+wheel → horizontal scroll for every scroll area
  (tables, trees, text edits). Widgets with their own wheel meaning
  (waveform canvas) aren't scroll areas, untouched."""

  def eventFilter(self, obj, ev):
    if ev.type() != QEvent.Wheel or not (ev.modifiers() & Qt.ShiftModifier):
      return False
    area = obj if isinstance(obj, QAbstractScrollArea) else obj.parent()
    if isinstance(area, QAbstractScrollArea) and _shift_hscroll(area, ev):
      return True
    return False

_filter = None  # module-level ref keeps the filter alive for the app lifetime

def install_shift_hscroll(app):
  global _filter
  if _filter is None:
    _filter = _ShiftHScrollFilter(app)
    app.installEventFilter(_filter)
