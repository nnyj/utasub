"""Single source for UI colors, fonts and app styling (dark theme).
Every color literal lives here; timeline/picker/log import from it."""
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
  QColor, QFont, QPalette, QIcon, QPixmap, QPainter, QPen, QPainterPath,
)

# --- greys: two levels only, everything muted picks one ---
GRAY = QColor(170, 170, 175)       # readable secondary text
DIM = QColor(130, 130, 140)        # hints, disabled-ish, struck credits

# --- surfaces ---
BG = QColor(30, 30, 35)            # canvas background
LANE = QColor(25, 25, 30)          # cue lane background
SURFACE = QColor(45, 45, 50)       # window / toolbar
SURFACE_HI = QColor(58, 58, 68)    # hover fill
BASE = QColor(37, 37, 42)          # text/entry background
BORDER = QColor(60, 60, 70)        # 1px separators, control outlines
ACCENT = QColor(70, 130, 220)      # selection / focus
TEXT = QColor(220, 220, 220)
CENTER_LINE = BORDER               # waveform center line
WAVE_PEAK = QColor(70, 160, 70)    # min/max peaks (dark green)
WAVE_RMS = QColor(120, 220, 120)   # inner RMS band (bright green)
ONSET = QColor(255, 200, 50, 60)   # onset marker (yellow)
REGION_MARKER = QColor(255, 100, 100, 120)  # region boundary (red dashed)
PLAYHEAD = QColor(255, 80, 80, 200)         # playhead line (red)
RULER = GRAY                       # time ruler major ticks/labels
RULER_MINOR = QColor(95, 95, 105)  # unlabeled minor ticks
RULER_BAND = QColor(38, 38, 45)    # ruler strip background (region-drag zone)
CUE_TEXT = TEXT                    # cue block label
CUE_ROMAJI = DIM                   # cue block romaji second line
EMPTY_HINT = DIM                   # canvas empty-state hint
SELECTED_BORDER = QColor(255, 255, 255, 220)  # selected cue border
PLAYING_BORDER = QColor(255, 210, 130, 230)  # accent border on cue under playhead

# --- confidence block colors (defined here, re-exported by timeline) ---
# amber: a cue or region the tool is least sure of, not an error
CONF_LOW = QColor(235, 175, 70)
CONF_LOW_BG = QColor(200, 140, 40, 55)

CONF_COLORS = {
  "lrc": QColor(60, 180, 80, 80),
  "fa": QColor(60, 100, 200, 80),
  "coarse": QColor(200, 140, 40, 80),
  "interpolated": QColor(200, 140, 40, 80),
  "provisional": QColor(150, 150, 150, 40),
  None: QColor(160, 160, 160, 60),
}

CONF_BORDER = {
  "lrc": QColor(60, 180, 80, 180),
  "fa": QColor(60, 100, 200, 180),
  "coarse": QColor(200, 140, 40, 180),
  "interpolated": QColor(200, 140, 40, 180),
  "provisional": QColor(150, 150, 150, 90),
  None: QColor(160, 160, 160, 140),
}

# --- picker / grid ---
PASTE_ROW = QColor(90, 150, 240)   # paste-row accent (readable blue on dark)
ROMAJI_CYAN = QColor(100, 210, 220)  # romaji interleave in preview
CREDIT_STRIKE = DIM                # struck-through credit lines
PLAYING_ROW = QColor(255, 210, 130, 40)  # grid row under playhead

# --- log / hints (stylesheet fragments) ---
LOG_BG = "#1e1e1e"
LOG_FG = "#ccc"
HINT_GRAY = DIM.name()

# log line coloring (by line shape, see gui/log.py)
LOG_HEADER = QColor(120, 175, 255)   # === section headers
LOG_WRITE = QColor(120, 200, 130)    # -> written files
LOG_WARN = QColor(225, 165, 95)      # errors / skips / misses
LOG_DETAIL = QColor(165, 165, 175)   # indented detail lines

# --- play/stop accents ---
PLAY_GREEN = "#4caf50"
STOP_RED = "#e05555"

# --- semantic toolbar-icon tints (neutral actions keep CUE_TEXT) ---
ICON_DESTRUCTIVE = "#e05555"  # delete, cleanup
ICON_WRITE = "#5cb85c"        # save, export, embed (produce a file)
ICON_ALIGN = "#5a90d8"        # align region / all
ICON_LRC = "#4ac06a"          # use LRC timestamps as-is (no alignment)
ICON_ASR = "#c08adc"          # transcribe (generate ASR)

# --- app-wide theming ---

FONT_PT = 10.5     # base UI size in points, Qt scales points by DPI itself
GROOVE = QColor(35, 35, 42)        # scrollbar groove
HANDLE = QColor(106, 106, 122)     # scrollbar handle
DISABLED = QColor(118, 118, 126)

def font_pt(delta=0.0):
  """UI size in points. Points are a physical unit and Qt scales them by the
  screen DPI itself, so no DPI division here."""
  return FONT_PT + delta

def canvas_font(delta=-1.0):
  """Waveform/ruler font, one step under the UI size."""
  f = QFont("Segoe UI")
  f.setPointSizeF(font_pt(delta=delta))
  return f

def apply_theme(app):
  """Fusion style + dark QPalette + font token + minimal app stylesheet."""
  app.setStyle("Fusion")

  f = app.font()
  f.setPointSizeF(font_pt())
  app.setFont(f)

  pal = QPalette()
  pal.setColor(QPalette.Window, SURFACE)
  pal.setColor(QPalette.WindowText, TEXT)
  pal.setColor(QPalette.Base, BASE)
  pal.setColor(QPalette.AlternateBase, SURFACE)
  pal.setColor(QPalette.Text, TEXT)
  pal.setColor(QPalette.Button, SURFACE)
  pal.setColor(QPalette.ButtonText, TEXT)
  pal.setColor(QPalette.ToolTipBase, BASE)
  pal.setColor(QPalette.ToolTipText, TEXT)
  pal.setColor(QPalette.Highlight, ACCENT)
  pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
  pal.setColor(QPalette.Link, ACCENT)
  pal.setColor(QPalette.Disabled, QPalette.Text, DISABLED)
  pal.setColor(QPalette.Disabled, QPalette.ButtonText, DISABLED)
  app.setPalette(pal)

  css = {k: v.name() for k, v in (("hi", SURFACE_HI), ("bd", BORDER),
                                  ("gr", GROOVE), ("hd", HANDLE),
                                  ("ac", ACCENT), ("bg", BASE))}
  app.setStyleSheet(
    "QToolBar {{ border: 0; padding: 4px; spacing: 4px; }}"
    "QToolButton {{ padding: 4px; border-radius: 4px; }}"
    "QToolButton:hover {{ background: {hi}; }}"
    "QToolButton:focus, QPushButton:focus {{ border: 1px solid {ac}; }}"
    "QMenuBar {{ padding: 2px; }}"
    "QMenuBar::item:selected {{ background: {hi}; }}"
    "QToolTip {{ border: 1px solid {bd}; padding: 4px; }}"
    "QHeaderView::section {{ background: {bg}; border: 0;"
    " border-bottom: 1px solid {bd}; padding: 4px; }}"
    "QTableWidget, QTreeWidget {{ gridline-color: {bd};"
    " selection-background-color: {ac}; }}"
    "QTableWidget::item, QTreeWidget::item {{ padding: 2px 4px; }}"
    # hover reads as a 1px outline, so only the selection is filled
    "QTableWidget::item:hover, QTreeWidget::item:hover {{"
    " border: 1px solid {hd}; }}"
    "QLineEdit, QComboBox, QDoubleSpinBox {{ border: 1px solid {bd};"
    " border-radius: 4px; padding: 2px 4px; }}"
    "QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{"
    " border: 1px solid {ac}; }}"
    # scrollbars need a contrasting groove, else they read as empty space
    "QScrollBar:horizontal {{ background: {gr}; border: 1px solid {bd};"
    " border-radius: 5px; height: 12px; margin: 0 12px; }}"
    "QScrollBar:vertical {{ background: {gr}; border: 1px solid {bd};"
    " border-radius: 5px; width: 12px; margin: 12px 0; }}"
    "QScrollBar::handle {{ background: {hd}; border-radius: 4px; }}"
    "QScrollBar::handle:hover {{ background: {ac}; }}"
    "QScrollBar::handle:horizontal {{ min-width: 24px; }}"
    "QScrollBar::handle:vertical {{ min-height: 24px; }}"
    "QScrollBar::add-line, QScrollBar::sub-line {{ background: none; border: none; }}"
    "QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}"
    .format(**css))

# --- icons ---
# One hand-drawn 16px family so every toolbar/menu icon shares stroke weight,
# color and metaphor. Cue ops (add/delete/split/merge) all use the same cue-block
# shape; align ops use blocks over a ruler line.

ICON_PX = 16       # logical icon size
ICON_SCALE = 2     # rendered at 2x, tagged hi-dpi, for crisp downscale
STROKE = 1.4

# shared block geometry: cue block, ruler baseline
_BLOCK = (2.5, 4.5, 11.0, 7.0)  # x, y, w, h

def _stroke(p, color, width=STROKE):
  pen = QPen(QColor(color))
  pen.setWidthF(width)
  pen.setCapStyle(Qt.RoundCap)
  pen.setJoinStyle(Qt.RoundJoin)
  p.setPen(pen)
  p.setBrush(Qt.NoBrush)

def _tri(p, color, pts):
  path = QPainterPath()
  path.moveTo(*pts[0])
  path.lineTo(*pts[1])
  path.lineTo(*pts[2])
  path.closeSubpath()
  p.fillPath(path, QColor(color))

def _rot_arrow(p, color, clockwise):
  """3/4-circle rotation arrow with a tangential head (undo = ccw, redo = cw)."""
  import math
  cx, cy, r = 8.0, 8.0, 4.6
  start, span = (200, -250) if clockwise else (-20, 250)
  _stroke(p, color)
  p.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r), start * 16, span * 16)
  # head at the arc end, pointing along the arc direction
  a = math.radians(start + span)
  tipx, tipy = cx + r * math.cos(a), cy - r * math.sin(a)
  sgn = 1 if span > 0 else -1
  dx, dy = -sgn * math.sin(a), -sgn * math.cos(a)
  px, py = -dy, dx  # perpendicular
  _tri(p, color, [
    (tipx + dx * 2.4, tipy + dy * 2.4),
    (tipx - dx * 1.2 + px * 2.1, tipy - dy * 1.2 + py * 2.1),
    (tipx - dx * 1.2 - px * 2.1, tipy - dy * 1.2 - py * 2.1),
  ])

def _block(p, color, x=None, y=None, w=None, h=None):
  bx, by, bw, bh = _BLOCK
  _stroke(p, color)
  p.drawRoundedRect(QRectF(x if x is not None else bx, y if y is not None else by,
                           w if w is not None else bw, h if h is not None else bh), 1.5, 1.5)

def _d_open(p, c):
  _stroke(p, c)
  path = QPainterPath()
  path.moveTo(2, 12.5)
  path.lineTo(2, 4)
  path.lineTo(6.2, 4)
  path.lineTo(7.6, 6)
  path.lineTo(14, 6)
  path.lineTo(14, 12.5)
  path.closeSubpath()
  p.drawPath(path)

def _d_save(p, c):
  _stroke(p, c)
  p.drawRoundedRect(QRectF(2.5, 2.5, 11, 11), 1.5, 1.5)
  p.drawRect(QRectF(5.5, 2.5, 5, 3.5))   # shutter
  p.drawRect(QRectF(4.5, 9, 7, 4.5))     # label

def _d_export(p, c):
  """Subtitle page with an arrow leaving it (write SRT)."""
  _page_out(p, c)
  for y in (5.5, 8, 10.5):  # subtitle lines
    p.drawLine(QPointF(4, y), QPointF(8, y))

def _page_out(p, c):
  """Shared page + outgoing arrow, the export family's base shape."""
  _stroke(p, c)
  p.drawRect(QRectF(2, 2.5, 8, 11))
  p.drawLine(QPointF(9, 11.5), QPointF(13.5, 11.5))
  _tri(p, c, [(15, 11.5), (12.2, 9.5), (12.2, 13.5)])

def _d_add(p, c):
  _block(p, c)
  _stroke(p, c)
  p.drawLine(QPointF(8, 5.8), QPointF(8, 10.2))
  p.drawLine(QPointF(5.8, 8), QPointF(10.2, 8))

def _d_delete(p, c):
  """Trash can: destructive, so it breaks the cue-block family on purpose."""
  _stroke(p, c)
  p.drawLine(QPointF(2.5, 4.5), QPointF(13.5, 4.5))       # lid
  p.drawLine(QPointF(6.5, 3), QPointF(9.5, 3))            # handle
  path = QPainterPath()                                    # body
  path.moveTo(4, 4.5)
  path.lineTo(4.9, 13.5)
  path.lineTo(11.1, 13.5)
  path.lineTo(12, 4.5)
  p.drawPath(path)
  for x in (6.6, 9.4):                                     # ribs
    p.drawLine(QPointF(x, 6.5), QPointF(x + 0.15, 11.5))

def _d_split(p, c):
  """One block cut in two, halves pushed apart."""
  _block(p, c, x=1.5, w=5)
  _block(p, c, x=9.5, w=5)
  _stroke(p, c)
  p.setPen(QPen(QColor(c), STROKE, Qt.DotLine))
  p.drawLine(QPointF(8, 3), QPointF(8, 13))

def _d_merge(p, c):
  """Two blocks pulled together into one span."""
  _block(p, c, x=0.5, w=4.5)
  _block(p, c, x=11, w=4.5)
  _tri(p, c, [(9.6, 8), (6.4, 5.6), (6.4, 10.4)])
  _tri(p, c, [(6.4, 8), (9.6, 5.6), (9.6, 10.4)])

def _align_icon(p, c, filled):
  """Three cue blocks over a timeline baseline; `filled` marks the targets."""
  _stroke(p, c)
  p.drawLine(QPointF(1, 13.5), QPointF(15, 13.5))
  for i, x in enumerate((1.0, 6.0, 11.0)):
    rect = QRectF(x, 3.5, 4, 6.5)
    if i in filled:
      p.fillRect(rect, QColor(c))
    else:
      _stroke(p, c)
      p.drawRect(rect)
    p.fillRect(QRectF(x + 1.5, 11.0, 1, 1.6), QColor(c))  # tick to baseline

def _d_align_region(p, c):
  _align_icon(p, c, filled={1})

def _d_align_all(p, c):
  _align_icon(p, c, filled={0, 1, 2})

def _d_lrc_as_is(p, c):
  """Clock over two cue blocks: LRC timestamps taken as they are."""
  _stroke(p, c)
  p.drawEllipse(QRectF(1.5, 1.5, 7, 7))
  p.drawLine(QPointF(5, 5), QPointF(5, 3.2))    # hands
  p.drawLine(QPointF(5, 5), QPointF(6.8, 5))
  for x in (1.5, 8.5):
    p.drawRoundedRect(QRectF(x, 10, 6, 4), 1.2, 1.2)

def _d_export_ass(p, c):
  """Export page marked with an A: styled .ass output."""
  _page_out(p, c)
  p.drawLine(QPointF(4, 9), QPointF(6, 4))      # A
  p.drawLine(QPointF(6, 4), QPointF(8, 9))
  p.drawLine(QPointF(4.9, 7), QPointF(7.1, 7))

# --- region status marks (St column) ---

STATUS_COLORS = {
  "setlist": QColor(235, 195, 90),    # from a known setlist
  "found": QColor(110, 200, 120),     # confident match
  "ambiguous": QColor(220, 165, 70),  # weak match, needs a look
  "none": DIM,                        # nothing found
}

def _d_st_setlist(p, c):
  import math
  path = QPainterPath()
  for i in range(10):
    a = math.radians(-90 + i * 36)
    r = 6.5 if i % 2 == 0 else 2.8
    x, y = 8 + r * math.cos(a), 8 + r * math.sin(a)
    path.lineTo(x, y) if i else path.moveTo(x, y)
  path.closeSubpath()
  p.fillPath(path, QColor(c))

def _d_st_found(p, c):
  p.setBrush(QColor(c))
  p.setPen(Qt.NoPen)
  p.drawEllipse(QRectF(3.5, 3.5, 9, 9))

def _d_st_ambiguous(p, c):
  _stroke(p, c, 1.8)
  p.drawEllipse(QRectF(3.5, 3.5, 9, 9))

def _d_st_none(p, c):
  _stroke(p, c, 1.8)
  p.drawLine(QPointF(4, 8), QPointF(12, 8))

def status_icon(status):
  """Region status mark for the St column. Unknown/None → blank icon."""
  if status not in STATUS_COLORS:
    return QIcon()
  return icon(f"st_{status}", STATUS_COLORS[status])

def _d_play(p, c):
  _tri(p, c, [(4.5, 3), (13, 8), (4.5, 13)])

def _d_stop(p, c):
  p.fillRect(QRectF(4, 4, 8, 8), QColor(c))

def _d_fit(p, c):
  _stroke(p, c)
  p.drawLine(QPointF(2, 3.5), QPointF(2, 12.5))
  p.drawLine(QPointF(14, 3.5), QPointF(14, 12.5))
  p.drawLine(QPointF(4.5, 8), QPointF(11.5, 8))
  _tri(p, c, [(3.5, 8), (6.2, 5.8), (6.2, 10.2)])
  _tri(p, c, [(12.5, 8), (9.8, 5.8), (9.8, 10.2)])

def _d_transcribe(p, c):
  """Microphone: generate ASR transcript from audio."""
  _stroke(p, c)
  p.drawRoundedRect(QRectF(6, 2.2, 4, 7), 2, 2)          # capsule
  p.drawArc(QRectF(4, 4.5, 8, 7), 200 * 16, 140 * 16)    # cradle
  p.drawLine(QPointF(8, 11.2), QPointF(8, 13.5))         # stand
  p.drawLine(QPointF(5.5, 13.5), QPointF(10.5, 13.5))    # base

def _d_cleanup(p, c):
  """Broom sweeping: clear stray files."""
  _stroke(p, c)
  p.drawLine(QPointF(13, 2.8), QPointF(7.6, 8.2))        # handle
  _tri(p, c, [(7.6, 8.2), (10.2, 10.4), (5.4, 12.6)])    # brush wedge
  for dx, dy in ((0.4, 0.9), (1.0, 0.4), (-0.2, 1.3)):   # bristle tips
    p.drawLine(QPointF(5.4 + dx * 2, 12.6 + dy), QPointF(5.4 + dx * 2 + 0.6, 13.8 + dy))

def _d_embed(p, c):
  """Film frame with a subtitle bar: mux SRT into the video."""
  _stroke(p, c)
  p.drawRoundedRect(QRectF(2, 3, 12, 10), 1.2, 1.2)      # frame
  for y in (4.4, 7.0, 9.6):                              # sprocket holes
    p.fillRect(QRectF(3.0, y, 1.2, 1.4), QColor(c))
    p.fillRect(QRectF(11.8, y, 1.2, 1.4), QColor(c))
  p.fillRect(QRectF(5.5, 10.2, 5, 1.4), QColor(c))       # subtitle bar

_DRAW = {
  "open": _d_open, "save": _d_save, "export": _d_export,
  "transcribe": _d_transcribe, "cleanup": _d_cleanup, "embed": _d_embed,
  "undo": lambda p, c: _rot_arrow(p, c, clockwise=False),
  "redo": lambda p, c: _rot_arrow(p, c, clockwise=True),
  "add": _d_add, "delete": _d_delete, "split": _d_split, "merge": _d_merge,
  "align_region": _d_align_region, "align_all": _d_align_all,
  "lrc_as_is": _d_lrc_as_is, "export_ass": _d_export_ass,
  "play": _d_play, "stop": _d_stop, "fit": _d_fit,
  "st_setlist": _d_st_setlist, "st_found": _d_st_found,
  "st_ambiguous": _d_st_ambiguous, "st_none": _d_st_none,
}

def _draw_app_mark(p, s):
  """App mark in an s x s box: waveform bars over a subtitle bar, dark tile."""
  p.setRenderHint(QPainter.Antialiasing, True)
  p.scale(s / 64.0, s / 64.0)
  p.setPen(Qt.NoPen)
  p.setBrush(QColor(34, 34, 40))
  p.drawRoundedRect(QRectF(1, 1, 62, 62), 12, 12)
  # waveform: symmetric bars around the middle, tallest in the center
  heights = [8, 15, 24, 30, 22, 32, 18, 11]
  p.setBrush(WAVE_RMS)
  for i, h in enumerate(heights):
    x = 8 + i * 6.4
    p.drawRoundedRect(QRectF(x, 26 - h / 2, 3.6, h), 1.8, 1.8)
  # subtitle bar
  p.setBrush(QColor(235, 235, 240))
  p.drawRoundedRect(QRectF(10, 44, 44, 6), 3, 3)
  p.setBrush(QColor(150, 150, 160))
  p.drawRoundedRect(QRectF(18, 54, 28, 5), 2.5, 2.5)

def app_icon():
  """Window/taskbar icon, rendered at the sizes Windows picks from."""
  ic = QIcon()
  for s in (16, 24, 32, 48, 64, 128, 256):
    pm = QPixmap(s, s)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    _draw_app_mark(p, s)
    p.end()
    ic.addPixmap(pm)
  return ic

def icon(name, color=CUE_TEXT):
  """Drawn 16px icon by name. Unknown name → null icon."""
  draw = _DRAW.get(name)
  if draw is None:
    return QIcon()
  pm = QPixmap(ICON_PX * ICON_SCALE, ICON_PX * ICON_SCALE)
  pm.fill(Qt.transparent)
  p = QPainter(pm)
  p.setRenderHint(QPainter.Antialiasing, True)
  p.scale(ICON_SCALE, ICON_SCALE)
  draw(p, color)
  p.end()
  pm.setDevicePixelRatio(ICON_SCALE)
  return QIcon(pm)
