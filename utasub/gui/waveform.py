"""Waveform canvas: min/max+RMS render, cue block drag/edit, region handles, ruler."""
import math
from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
  QColor, QPainter, QPen, QFont, QFontMetrics,
  QWheelEvent, QMouseEvent,
)
from PySide6.QtWidgets import QWidget

from ..core.align import Cue
from . import theme
from .theme import CONF_COLORS, CONF_BORDER
from .timeline_util import (
  romaji_for, find_onsets, UndoStack, fmt_tick, TICK_STEPS,
  DRAG_NONE, DRAG_MOVE, DRAG_LEFT, DRAG_RIGHT, DRAG_PAN, DRAG_REGION,
  EDGE_THRESH_PX, NEIGHBOR_SNAP_PX, REGION_THRESH_PX, MIN_REGION_S,
)


class WaveformCanvas(QWidget):
  """Custom paint widget: waveform + cue block lane."""

  cue_changed = Signal()    # emitted after any cue edit
  selection_changed = Signal(int)  # emitted with selected cue index
  seek_requested = Signal(float)   # click-scrub time
  region_changed = Signal(int)     # emitted with region index after boundary drag
  view_changed = Signal()   # emitted on zoom/pan/fit/zoom_to_span/follow shift
  playing_cue_changed = Signal(int)  # cue index under the playhead, -1 when none
  playing_region_changed = Signal(int)  # region index under the playhead, -1 when none

  CUE_LANE_H = 44          # px for cue lane (fits 2 text lines: text + romaji)
  RULER_H = 18             # px for the time ruler strip
  BOTTOM_MARGIN = 10       # breathing room under the ruler labels
  MIN_WAVEFORM_H = 40      # waveform never collapses below this

  def __init__(self, parent=None):
    super().__init__(parent)
    self.setMinimumHeight(120)  # MIN_WAVEFORM_H + lane + ruler + margin
    self.setMouseTracking(True)
    self.setFocusPolicy(Qt.StrongFocus)
    self.setToolTip(
      "Drag = move · edges = resize · Shift+drag = ripple · Alt = no snap\n"
      "Shift+wheel = pan · region edges drag in the ruler band")

    # data
    self._audio = None      # raw mono float32; None → empty waveform area
    self._sr = 16000
    self._pyr_factors = (16, 256, 4096)
    self._pyramid = {}      # factor -> (mins, maxs, mean_square) arrays
    self._col_cache_key = None
    self._col_cache = None  # (col_min, col_max, col_rms) per pixel
    self.provisional = False
    self.cues = []          # list of Cue
    self.onsets = []        # onset times in seconds
    self.regions = []       # Region objects for vertical markers
    self.duration = 0.0     # audio duration in seconds

    # view state
    self.view_start = 0.0   # left edge time
    self.view_end = 10.0    # right edge time
    self.selected = -1      # selected cue index (into self.cues)
    self.playhead = 0.0     # current playback position
    self.playing_cue = -1   # cue index under the playhead (drives grid row glow)
    self.playing_region = -1  # region index under the playhead (drives region-list glow)
    self.follow = True      # follow playhead during playback (page-flip view)

    # drag state
    self._drag_mode = DRAG_NONE
    self._drag_cue = -1
    self._drag_start_x = 0
    self._drag_orig_start = 0.0
    self._drag_orig_end = 0.0
    self._drag_ripple = False
    self._drag_snap_suppress = False
    self._drag_orig_following = []  # [(idx, orig_start, orig_end), ...] for ripple
    self._drag_region_idx = -1     # which region is being dragged
    self._drag_region_edge = None  # 'start' or 'end'

    # undo
    self.undo_stack = UndoStack()
    self._undo_snapshot = None  # (idx, Cue) taken on drag press, committed on release

    # dirty flag
    self._dirty = False

    self._font = QFont("Segoe UI", 8)
    self._label_fm = QFontMetrics(self._font)

  # --- waveform source ---

  def set_audio(self, audio, sr=16000):
    """Store raw mono float32 audio, compute onsets + min/max pyramid."""
    import numpy as np
    self._col_cache_key = None
    self._col_cache = None
    if audio is None or len(audio) == 0:
      self._audio = None
      self._pyramid = {}
      self.duration = 0.0
      return
    self._audio = np.ascontiguousarray(audio, dtype=np.float32)
    self._sr = sr
    self.duration = len(self._audio) / sr
    self.onsets = find_onsets(self._audio, sr)
    self._build_pyramid()

  def _build_pyramid(self):
    """Downsample pyramid: per factor, block min/max/mean-square arrays."""
    import numpy as np
    self._pyramid = {}
    a = self._audio
    for f in self._pyr_factors:
      n = len(a) // f
      if n == 0:
        break
      block = a[:n * f].reshape(n, f)
      self._pyramid[f] = (
        block.min(axis=1), block.max(axis=1),
        (block.astype(np.float64) ** 2).mean(axis=1),
      )

  def _pick_source(self, samples_per_pixel):
    """Choose finest source (raw or pyramid level) with >=1 elem/pixel.
    Returns (mins, maxs, mean_square|None, rate)."""
    best = None
    for f in self._pyr_factors:
      p = self._pyramid.get(f)
      if p is not None and samples_per_pixel / f >= 1.0:
        best = (f, p)
    if best is None:
      return self._audio, self._audio, None, self._sr  # raw
    f, (mn, mx, ms) = best
    return mn, mx, ms, self._sr / f

  def _compute_columns(self, vs, ve, width):
    """Per-pixel (min, max, rms) over visible span, vectorized + cached."""
    key = (round(vs, 4), round(ve, 4), width)
    if self._col_cache_key == key and self._col_cache is not None:
      return self._col_cache
    import numpy as np
    span = max(ve - vs, 1e-6)
    spp = span / max(width, 1) * self._sr
    src_min, src_max, src_ms, rate = self._pick_source(spp)
    n_src = len(src_min)
    i0 = vs * rate
    i1 = ve * rate
    lo = max(int(math.floor(i0)), 0)
    hi = min(int(math.ceil(i1)) + 1, n_src)
    if hi <= lo:
      empty = np.zeros(width, np.float32)
      self._col_cache_key = key
      self._col_cache = (empty, empty.copy(), empty.copy())
      return self._col_cache
    vmin = src_min[lo:hi]
    vmax = src_max[lo:hi]
    vms = src_ms[lo:hi] if src_ms is not None else vmin.astype(np.float64) ** 2
    m = len(vmin)
    edges = np.linspace(i0 - lo, i1 - lo, width + 1)
    idx = np.clip(np.floor(edges).astype(np.int64), 0, m - 1)
    starts = idx[:-1]
    col_min = np.minimum.reduceat(vmin, starts).astype(np.float32)
    col_max = np.maximum.reduceat(vmax, starts).astype(np.float32)
    ms_sum = np.add.reduceat(vms, starts)
    counts = np.diff(np.append(starts, m)).astype(np.float64)
    counts[counts < 1] = 1.0
    col_rms = np.sqrt(ms_sum / counts).astype(np.float32)
    self._col_cache_key = key
    self._col_cache = (col_min, col_max, col_rms)
    return self._col_cache

  # --- coordinate helpers ---

  def time_to_x(self, t):
    w = self.width()
    span = max(self.view_end - self.view_start, 0.001)
    return (t - self.view_start) / span * w

  def x_to_time(self, x):
    w = max(self.width(), 1)
    span = max(self.view_end - self.view_start, 0.001)
    return self.view_start + x / w * span

  def cue_rect(self, idx):
    """Return (x, y, w, h) for cue block in the lane."""
    if idx < 0 or idx >= len(self.cues):
      return (0, 0, 0, 0)
    c = self.cues[idx]
    x1 = self.time_to_x(c.start)
    x2 = self.time_to_x(c.end)
    y = self._cue_lane_y()
    return (x1, y, x2 - x1, self.CUE_LANE_H)

  def _waveform_h(self):
    """Waveform takes whatever the lane + ruler + bottom margin do not, so the
    ruler labels stay fully visible even at the minimum canvas height."""
    rest = self.CUE_LANE_H + self.RULER_H + self.BOTTOM_MARGIN + 4
    return max(self.MIN_WAVEFORM_H, self.height() - rest)

  def _cue_lane_y(self):
    return self._waveform_h() + 2

  def _ruler_y(self):
    return self._cue_lane_y() + self.CUE_LANE_H + 2

  def _tick_steps(self, span, width):
    """(major, minor) ruler steps for the visible span: major labels ~70px
    apart, minor ticks the densest ladder step still >=8px apart."""
    ideal = max(span, 1e-6) / max(width / 70.0, 1.0)
    major = next((s for s in TICK_STEPS if s >= ideal), TICK_STEPS[-1])
    minor = next((s for s in TICK_STEPS
                  if s < major and s / max(span, 1e-6) * width >= 8), major)
    return major, minor

  # --- hit testing ---

  def _edge_tol(self, i, selected):
    """Edge grab tolerance for cue i. The selected cue keeps the full band so
    its edges always win; other cues give up tolerance on narrow blocks so a
    body click can still select them."""
    if selected:
      return EDGE_THRESH_PX
    rw = self.cue_rect(i)[2]
    return min(EDGE_THRESH_PX, max(2.0, rw / 3.0))

  def _edge_hits(self, i, x, tol):
    """Edge candidates for cue i at x: [(distance, inside, mode), ...].
    `inside` marks the edge whose own block contains x, which breaks the tie
    when two cues share a boundary (cue.end == next.start)."""
    rx, _, rw, _ = self.cue_rect(i)
    out = []
    if abs(x - rx) <= tol:
      out.append((abs(x - rx), x >= rx, DRAG_LEFT))
    if abs(x - (rx + rw)) <= tol:
      out.append((abs(x - (rx + rw)), x <= rx + rw, DRAG_RIGHT))
    return out

  def _hit_cue(self, x, y):
    """Returns (cue_idx, drag_mode) or (-1, DRAG_NONE).
    Order: the selected cue's own edges, then the nearest edge of any cue
    (preferring the one whose block x sits inside), then a body move."""
    lane_y = self._cue_lane_y()
    if y < lane_y or y > lane_y + self.CUE_LANE_H:
      return -1, DRAG_NONE
    sel = self.selected
    if 0 <= sel < len(self.cues):
      hits = self._edge_hits(sel, x, self._edge_tol(sel, True))
      if hits:
        return sel, min(hits, key=lambda h: (h[0], not h[1]))[2]
    # nearest edge across cues; ties at a shared boundary go to the block the
    # cursor is actually inside, so the neighbour is never grabbed
    best = None
    for i in range(len(self.cues)):
      for dist, inside, mode in self._edge_hits(i, x, self._edge_tol(i, False)):
        key = (dist, not inside)
        if best is None or key < best[0]:
          best = (key, i, mode)
    if best is not None:
      return best[1], best[2]
    for i in range(len(self.cues) - 1, -1, -1):
      rx, _, rw, _ = self.cue_rect(i)
      if rx <= x <= rx + rw:
        return i, DRAG_MOVE
    return -1, DRAG_NONE

  # --- region hit testing ---

  def _hit_region(self, x, y=None):
    """Returns (region_idx, 'start'|'end') or (-1, None) if no hit.
    Grabbable only inside the ruler strip (y in the ruler band), so plain
    waveform clicks seek."""
    if y is not None and y < self._ruler_y():
      return -1, None
    for i, r in enumerate(self.regions):
      sx = int(self.time_to_x(r.start))
      ex = int(self.time_to_x(r.end))
      if abs(x - sx) <= REGION_THRESH_PX:
        return i, 'start'
      if abs(x - ex) <= REGION_THRESH_PX:
        return i, 'end'
    return -1, None

  def move_region_bound(self, idx, edge, new_time):
    """Move region boundary, clamped by neighbors, duration, min length."""
    if idx < 0 or idx >= len(self.regions):
      return
    r = self.regions[idx]
    new_time = max(0.0, new_time)
    if self.duration > 0:
      new_time = min(new_time, self.duration)
    if edge == 'start':
      # clamp: can't go past own end minus min, can't go before prev region end
      new_time = min(new_time, r.end - MIN_REGION_S)
      if idx > 0:
        new_time = max(new_time, self.regions[idx - 1].end)
      r.start = round(max(0.0, new_time), 3)
    else:
      # clamp: can't go before own start plus min, can't go past next region start
      new_time = max(new_time, r.start + MIN_REGION_S)
      if idx + 1 < len(self.regions):
        new_time = min(new_time, self.regions[idx + 1].start)
      if self.duration > 0:
        new_time = min(new_time, self.duration)
      r.end = round(new_time, 3)
    self.region_changed.emit(idx)
    self.update()

  # --- snap ---

  def snap_to_onset(self, t, suppress=False):
    """Snap time to nearest onset within 150ms. Alt suppresses."""
    if suppress or not self.onsets:
      return t
    nearest = min(self.onsets, key=lambda o: abs(o - t))
    if abs(nearest - t) <= 0.150:
      return nearest
    return t

  def snap_to_neighbor(self, t, bound, suppress=False):
    """Snap t onto a neighbouring cue's edge when within NEIGHBOR_SNAP_PX.
    Alt suppresses, same as onset snap."""
    if suppress:
      return t
    if abs(self.time_to_x(t) - self.time_to_x(bound)) <= NEIGHBOR_SNAP_PX:
      return bound
    return t

  # --- cue editing ---

  def _enforce_non_decreasing(self, from_idx=0):
    """Ensure starts stay non-decreasing from from_idx onward."""
    for j in range(max(1, from_idx), len(self.cues)):
      if self.cues[j].start < self.cues[j - 1].start:
        diff = self.cues[j - 1].start + 0.05 - self.cues[j].start
        c = self.cues[j]
        self.cues[j] = replace(c, start=c.start + diff, end=c.end + diff)

  def _mark_dirty(self):
    self._dirty = True
    self.cue_changed.emit()
    self.update()

  @property
  def dirty(self):
    return self._dirty

  def clear_dirty(self):
    self._dirty = False

  # --- structural edits (insert/remove/split/merge) ---

  def insert_cue(self, idx, cue):
    """Insert cue at idx, push undo."""
    self.cues.insert(idx, cue)
    self.undo_stack.push(('insert', idx, replace(cue)))
    self._mark_dirty()

  def remove_cue(self, idx):
    """Remove cue at idx, push undo. Returns removed cue."""
    if idx < 0 or idx >= len(self.cues):
      return None
    removed = self.cues.pop(idx)
    self.undo_stack.push(('remove', idx, replace(removed)))
    if self.selected >= len(self.cues):
      self.selected = len(self.cues) - 1
    self._mark_dirty()
    return removed

  def split_cue(self, idx, split_time):
    """Split cue at split_time into two cues. Text duplicated to both halves."""
    if idx < 0 or idx >= len(self.cues):
      return
    c = self.cues[idx]
    if split_time <= c.start + 0.05 or split_time >= c.end - 0.05:
      return  # too close to edges
    old = replace(c)
    self.cues[idx] = replace(c, end=split_time)
    new_cue = replace(c, start=split_time)
    self.cues.insert(idx + 1, new_cue)
    self.undo_stack.push(('split', idx, old, replace(self.cues[idx]),
                          replace(new_cue)))
    self._mark_dirty()

  def merge_cue(self, idx):
    """Merge cue[idx] with cue[idx+1]: concat text, span union."""
    if idx < 0 or idx + 1 >= len(self.cues):
      return
    a = self.cues[idx]
    b = self.cues[idx + 1]
    merged_text = f"{a.text} {b.text}".strip()
    merged = Cue(a.start, b.end, merged_text, a.confidence)
    old_a = replace(a)
    old_b = replace(b)
    self.cues[idx] = merged
    self.cues.pop(idx + 1)
    self.undo_stack.push(('merge', idx, old_a, old_b, replace(merged)))
    if self.selected >= len(self.cues):
      self.selected = len(self.cues) - 1
    self._mark_dirty()

  def _apply_offset(self, delta):
    """Shift every cue by delta seconds, clamped so nothing goes negative.
    Returns the delta actually applied."""
    if not self.cues or not delta:
      return 0.0
    delta = max(delta, -min(c.start for c in self.cues))
    if not delta:
      return 0.0
    self.cues = [replace(c, start=c.start + delta, end=c.end + delta)
                 for c in self.cues]
    return delta

  def offset_cues(self, delta):
    """User-facing cue offset: applies, pushes undo, marks dirty."""
    applied = self._apply_offset(delta)
    if applied:
      self.undo_stack.push(('offset', applied))
      self._mark_dirty()
    return applied

  # --- zoom / pan ---

  def zoom(self, factor, center_t=None):
    """Zoom view by factor around center_t."""
    if center_t is None:
      center_t = (self.view_start + self.view_end) / 2
    span = self.view_end - self.view_start
    new_span = max(0.5, min(span * factor, self.duration or 300))
    ratio = (center_t - self.view_start) / max(span, 0.001)
    self.view_start = center_t - new_span * ratio
    self.view_end = self.view_start + new_span
    self._clamp_view()
    self.view_changed.emit()
    self.update()

  def pan(self, delta_t):
    self.view_start += delta_t
    self.view_end += delta_t
    self._clamp_view()
    self.view_changed.emit()
    self.update()

  def _clamp_view(self):
    span = self.view_end - self.view_start
    if self.view_start < 0:
      self.view_start = 0.0
      self.view_end = span
    if self.duration > 0 and self.view_end > self.duration:
      self.view_end = self.duration
      self.view_start = max(0.0, self.view_end - span)

  def fit_all(self):
    """Zoom to show entire duration."""
    self.view_start = 0.0
    self.view_end = max(self.duration, 1.0)
    self.view_changed.emit()
    self.update()

  def zoom_to_span(self, start, end):
    """Zoom/scroll to show [start, end] with 5% padding."""
    span = end - start
    pad = span * 0.05
    self.view_start = max(0.0, start - pad)
    self.view_end = end + pad
    self._clamp_view()
    self.view_changed.emit()
    self.update()

  # --- playhead ---

  def set_playhead(self, t):
    """Set playhead; when follow is on and t leaves the visible span, page-flip
    the view so t lands ~10% from the left, keeping span. Emits view_changed."""
    self.playhead = t
    idx = self._cue_at(t)
    if idx != self.playing_cue:
      self.playing_cue = idx
      self.playing_cue_changed.emit(idx)
    ridx = self._region_at(t)
    if ridx != self.playing_region:
      self.playing_region = ridx
      self.playing_region_changed.emit(ridx)
    if self.follow and (t > self.view_end or t < self.view_start):
      span = self.view_end - self.view_start
      self.view_start = max(0.0, t - span * 0.1)
      self.view_end = self.view_start + span
      self._clamp_view()
      self.view_changed.emit()
    self.update()

  def _cue_at(self, t):
    """Index of the cue spanning time t, or -1. Cues are start-ordered."""
    if t <= 0:
      return -1
    for i, c in enumerate(self.cues):
      if c.start <= t <= c.end:
        return i
      if c.start > t:
        break
    return -1

  def _region_at(self, t):
    """Index of the region spanning time t, or -1."""
    if t <= 0:
      return -1
    for i, r in enumerate(self.regions):
      if r.start <= t <= r.end:
        return i
    return -1

  # --- mouse events ---

  def mousePressEvent(self, ev: QMouseEvent):
    x, y = ev.position().x(), ev.position().y()
    mods = ev.modifiers()
    self._drag_snap_suppress = bool(mods & Qt.AltModifier)
    self._drag_ripple = bool(mods & Qt.ShiftModifier)

    # region boundary drag (ruler strip only); ignored while provisional
    ridx, redge = self._hit_region(x, y)
    if ridx >= 0 and not self.provisional:
      self._drag_mode = DRAG_REGION
      self._drag_region_idx = ridx
      self._drag_region_edge = redge
      self._drag_start_x = x
      self.update()
      return

    idx, mode = self._hit_cue(x, y)
    if idx >= 0:
      # provisional: allow selection only, ignore drags/edits
      if self.provisional:
        self.selected = idx
        self.selection_changed.emit(idx)
        self.update()
        return
      self._drag_mode = mode
      self._drag_cue = idx
      self._drag_start_x = x
      c = self.cues[idx]
      self._drag_orig_start = c.start
      self._drag_orig_end = c.end
      if self._drag_ripple:
        self._drag_orig_following = [
          (j, self.cues[j].start, self.cues[j].end)
          for j in range(idx + 1, len(self.cues))
        ]
      self.selected = idx
      self.selection_changed.emit(idx)
      self._push_undo_snapshot(idx)
      self.update()
      return

    # click in waveform area or ruler = seek
    if y < self._cue_lane_y() or y >= self._ruler_y():
      t = self.x_to_time(x)
      self.seek_requested.emit(max(0.0, t))
      # start pan drag
      self._drag_mode = DRAG_PAN
      self._drag_start_x = x
      return

    # click in empty cue lane = deselect
    self.selected = -1
    self.selection_changed.emit(-1)
    self._drag_mode = DRAG_PAN
    self._drag_start_x = x
    self.update()

  def mouseMoveEvent(self, ev: QMouseEvent):
    x, y = ev.position().x(), ev.position().y()
    if self._drag_mode == DRAG_NONE:
      # update cursor: region markers (ruler strip) first
      ridx, _ = self._hit_region(x, y)
      if ridx >= 0:
        self.setCursor(Qt.SizeHorCursor)
        return
      idx, mode = self._hit_cue(x, y)
      if mode in (DRAG_LEFT, DRAG_RIGHT):
        self.setCursor(Qt.SizeHorCursor)
      elif mode == DRAG_MOVE:
        self.setCursor(Qt.OpenHandCursor)
      else:
        self.setCursor(Qt.ArrowCursor)
      return

    if self._drag_mode == DRAG_REGION:
      t = self.x_to_time(x)
      self.move_region_bound(self._drag_region_idx, self._drag_region_edge, t)
      return

    if self._drag_mode == DRAG_PAN:
      dx = x - self._drag_start_x
      dt = -dx / max(self.width(), 1) * (self.view_end - self.view_start)
      self.pan(dt)
      self._drag_start_x = x
      return

    # cue drag
    idx = self._drag_cue
    if idx < 0 or idx >= len(self.cues):
      return
    dt = self.x_to_time(x) - self.x_to_time(self._drag_start_x)

    if self._drag_mode == DRAG_MOVE:
      c = self.cues[idx]
      orig_dur = self._drag_orig_end - self._drag_orig_start
      new_start = self.snap_to_onset(
        self._drag_orig_start + dt, self._drag_snap_suppress)
      new_start = max(0.0, new_start)
      if idx > 0:
        new_start = self.snap_to_neighbor(
          new_start, self.cues[idx - 1].end, self._drag_snap_suppress)
      if idx + 1 < len(self.cues) and not self._drag_ripple:
        # snap trailing edge too; with ripple the next cue moves along, so
        # its start is not a fixed target
        snapped_end = self.snap_to_neighbor(
          new_start + orig_dur, self.cues[idx + 1].start,
          self._drag_snap_suppress)
        new_start = snapped_end - orig_dur
      if idx > 0:
        new_start = max(new_start, self.cues[idx - 1].start + 0.05)
      self.cues[idx] = replace(c, start=new_start, end=new_start + orig_dur)
      if self._drag_ripple:
        shift = new_start - self._drag_orig_start
        for j, os, oe in self._drag_orig_following:
          if j < len(self.cues):
            cj = self.cues[j]
            self.cues[j] = replace(cj, start=os + shift, end=oe + shift)
      self._enforce_non_decreasing(idx)

    elif self._drag_mode == DRAG_LEFT:
      new_start = self.snap_to_onset(
        self._drag_orig_start + dt, self._drag_snap_suppress)
      new_start = max(0.0, new_start)
      if idx > 0:
        prev_end = self.cues[idx - 1].end
        new_start = self.snap_to_neighbor(
          new_start, prev_end, self._drag_snap_suppress)
        # never resize into the previous cue; Alt bypasses, and skipped when
        # the pair already overlapped, else the edge would jump on first pixel
        if (not self._drag_snap_suppress
            and self._drag_orig_start >= prev_end - 1e-6):
          new_start = max(new_start, prev_end)
        new_start = max(new_start, self.cues[idx - 1].start + 0.05)
      new_start = min(new_start, self.cues[idx].end - 0.05)
      c = self.cues[idx]
      self.cues[idx] = replace(c, start=new_start)

    elif self._drag_mode == DRAG_RIGHT:
      new_end = self.snap_to_onset(
        self._drag_orig_end + dt, self._drag_snap_suppress)
      if idx + 1 < len(self.cues):
        next_start = self.cues[idx + 1].start
        new_end = self.snap_to_neighbor(
          new_end, next_start, self._drag_snap_suppress)
        if (not self._drag_snap_suppress
            and self._drag_orig_end <= next_start + 1e-6):
          new_end = min(new_end, next_start)
      new_end = max(self.cues[idx].start + 0.05, new_end)
      c = self.cues[idx]
      self.cues[idx] = replace(c, end=new_end)

    self._dirty = True
    self.update()

  def mouseReleaseEvent(self, ev: QMouseEvent):
    if self._drag_mode in (DRAG_MOVE, DRAG_LEFT, DRAG_RIGHT):
      self.commit_undo()  # finalize the snapshot taken on press into the undo stack
      self.cue_changed.emit()
    if self._drag_mode == DRAG_REGION:
      self.region_changed.emit(self._drag_region_idx)
    self._drag_mode = DRAG_NONE
    self._drag_cue = -1
    self._drag_region_idx = -1
    self._drag_region_edge = None
    self._drag_orig_following = []

  def wheelEvent(self, ev: QWheelEvent):
    delta = ev.angleDelta().y()
    if delta == 0:
      return
    if ev.modifiers() & Qt.ShiftModifier:
      # shift+wheel = horizontal scroll by 15% of the visible span
      span = self.view_end - self.view_start
      self.pan(-delta / 120.0 * span * 0.15)
      return
    factor = 0.8 if delta > 0 else 1.25
    center = self.x_to_time(ev.position().x())
    self.zoom(factor, center)

  # --- undo helpers ---

  def _push_undo_snapshot(self, idx):
    if idx < 0 or idx >= len(self.cues):
      return
    c = self.cues[idx]
    self._undo_snapshot = (idx, replace(c))

  def commit_undo(self):
    """Call after mouse release to finalize undo entry. A ripple drag carries
    the cues it dragged along, so one undo restores the whole move."""
    if self._undo_snapshot is None:
      return
    idx, old = self._undo_snapshot
    if idx < len(self.cues):
      new = self.cues[idx]
      if old.start != new.start or old.end != new.end:
        rippled = [(j, replace(self.cues[j], start=os, end=oe), replace(self.cues[j]))
                   for j, os, oe in self._drag_orig_following if j < len(self.cues)]
        self.undo_stack.push(('timing', idx, old, replace(new), rippled))
    self._undo_snapshot = None

  def do_undo(self):
    cmd = self.undo_stack.undo()
    if cmd is None:
      return
    kind = cmd[0]
    if kind == 'timing':
      idx, old = cmd[1], cmd[2]
      if idx < len(self.cues):
        self.cues[idx] = old
        for j, old_j, _new_j in (cmd[4] if len(cmd) > 4 else []):
          if j < len(self.cues):
            self.cues[j] = old_j
        self._mark_dirty()
    elif kind == 'insert':
      _, idx, cue = cmd
      if idx < len(self.cues):
        self.cues.pop(idx)
        if self.selected >= len(self.cues):
          self.selected = max(-1, len(self.cues) - 1)
        self._mark_dirty()
    elif kind == 'remove':
      _, idx, cue = cmd
      self.cues.insert(idx, cue)
      self._mark_dirty()
    elif kind == 'split':
      _, idx, old, first_half, second_half = cmd
      if idx + 1 < len(self.cues):
        self.cues.pop(idx + 1)
      if idx < len(self.cues):
        self.cues[idx] = old
      self._mark_dirty()
    elif kind == 'merge':
      _, idx, old_a, old_b, merged = cmd
      if idx < len(self.cues):
        self.cues[idx] = old_a
        self.cues.insert(idx + 1, old_b)
      self._mark_dirty()
    elif kind == 'offset':
      self._apply_offset(-cmd[1])
      self._mark_dirty()

  def do_redo(self):
    cmd = self.undo_stack.redo()
    if cmd is None:
      return
    kind = cmd[0]
    if kind == 'timing':
      idx, new = cmd[1], cmd[3]
      if idx < len(self.cues):
        self.cues[idx] = new
        for j, _old_j, new_j in (cmd[4] if len(cmd) > 4 else []):
          if j < len(self.cues):
            self.cues[j] = new_j
        self._mark_dirty()
    elif kind == 'insert':
      _, idx, cue = cmd
      self.cues.insert(idx, cue)
      self._mark_dirty()
    elif kind == 'remove':
      _, idx, cue = cmd
      if idx < len(self.cues):
        self.cues.pop(idx)
        if self.selected >= len(self.cues):
          self.selected = max(-1, len(self.cues) - 1)
        self._mark_dirty()
    elif kind == 'split':
      _, idx, old, first_half, second_half = cmd
      if idx < len(self.cues):
        self.cues[idx] = first_half
        self.cues.insert(idx + 1, second_half)
      self._mark_dirty()
    elif kind == 'merge':
      _, idx, old_a, old_b, merged = cmd
      if idx < len(self.cues) and idx + 1 < len(self.cues):
        self.cues[idx] = merged
        self.cues.pop(idx + 1)
      self._mark_dirty()
    elif kind == 'offset':
      self._apply_offset(cmd[1])
      self._mark_dirty()

  # --- paint ---

  def paintEvent(self, ev):
    p = QPainter(self)
    p.setRenderHint(QPainter.Antialiasing, False)
    w = self.width()
    h = self.height()

    # background
    p.fillRect(0, 0, w, h, theme.BG)

    wf_h = self._waveform_h()
    mid_y = wf_h // 2

    # center line (subtle)
    p.setPen(QPen(theme.CENTER_LINE, 1))
    p.drawLine(0, mid_y, w, mid_y)

    # waveform: raw-audio path (Audacity-style min/max + RMS band) takes precedence
    if self._audio is not None and len(self._audio):
      col_min, col_max, col_rms = self._compute_columns(
        self.view_start, self.view_end, w)
      amp = mid_y * 0.9
      # outer min/max peaks (dark green)
      p.setPen(QPen(theme.WAVE_PEAK, 1))
      for px in range(w):
        y_top = mid_y - int(col_max[px] * amp)
        y_bot = mid_y - int(col_min[px] * amp)
        p.drawLine(px, y_top, px, y_bot)
      # inner RMS band (brighter green)
      p.setPen(QPen(theme.WAVE_RMS, 1))
      for px in range(w):
        r = int(col_rms[px] * amp)
        p.drawLine(px, mid_y - r, px, mid_y + r)

    # onset markers (thin vertical lines in waveform area)
    p.setPen(QPen(theme.ONSET, 1))
    for ot in self.onsets:
      ox = int(self.time_to_x(ot))
      if 0 <= ox < w:
        p.drawLine(ox, 0, ox, wf_h)

    # cue lane background
    lane_y = self._cue_lane_y()
    p.fillRect(0, lane_y, w, self.CUE_LANE_H, theme.LANE)

    # empty-state hint (no cues, not provisional)
    if not self.cues and not self.provisional:
      p.setFont(self._font)
      p.setPen(QPen(theme.EMPTY_HINT, 1))
      p.drawText(0, lane_y, w, self.CUE_LANE_H, Qt.AlignCenter,
                 "No cues yet — assign lyrics per region, then Align all + Export")

    # cue blocks
    p.setFont(self._font)
    for i, c in enumerate(self.cues):
      x1 = int(self.time_to_x(c.start))
      x2 = int(self.time_to_x(c.end))
      bw = max(x2 - x1, 2)
      if self.provisional:
        conf = "provisional"
      else:
        conf = c.confidence if isinstance(c, Cue) else None
      fill = CONF_COLORS.get(conf, CONF_COLORS[None])
      border = CONF_BORDER.get(conf, CONF_BORDER[None])

      if i == self.selected:
        fill = QColor(fill.red(), fill.green(), fill.blue(), min(fill.alpha() + 60, 255))
        border = theme.SELECTED_BORDER

      # playback glow: cue currently under the playhead (only during playback)
      playing = self.playhead > 0 and c.start <= self.playhead <= c.end
      border_w = 1
      if playing:
        p.fillRect(x1, lane_y + 1, bw, self.CUE_LANE_H - 2, theme.PLAYING_FILL)
        if i != self.selected:
          border = theme.PLAYING_BORDER
        border_w = 2

      p.fillRect(x1, lane_y + 1, bw, self.CUE_LANE_H - 2, fill)
      p.setPen(QPen(border, border_w))
      p.drawRect(x1, lane_y + 1, bw, self.CUE_LANE_H - 2)

      # label text (+ romaji second line when block tall/wide enough)
      label = c.text[:40] if hasattr(c, 'text') else ""
      if bw > 20:
        max_chars = max(1, bw // 7)
        p.setPen(QPen(theme.CUE_TEXT, 1))
        p.drawText(x1 + 3, lane_y + 13, label[:max_chars])
        if self.CUE_LANE_H >= 30 and bw > 40:
          rom = romaji_for(c.text) if hasattr(c, 'text') else ""
          if rom:
            p.setPen(QPen(theme.CUE_ROMAJI, 1))
            p.drawText(x1 + 3, lane_y + 27, rom[:max_chars])

    # region vertical markers
    p.setPen(QPen(theme.REGION_MARKER, 2, Qt.DashLine))
    for r in self.regions:
      for t in (r.start, r.end):
        rx = int(self.time_to_x(t))
        if 0 <= rx < w:
          p.drawLine(rx, 0, rx, h)

    # playhead
    px = int(self.time_to_x(self.playhead))
    if 0 <= px < w:
      p.setPen(QPen(theme.PLAYHEAD, 2))
      p.drawLine(px, 0, px, h)

    # time ruler (bottom): also the only strip where region bounds are grabbable
    ruler_y = self._ruler_y()
    if ruler_y < h:
      p.fillRect(0, ruler_y, w, h - ruler_y, theme.RULER_BAND)
      span = self.view_end - self.view_start
      major, minor = self._tick_steps(span, w)
      # minor ticks: unlabeled, denser grid so long media still reads
      p.setPen(QPen(theme.RULER_MINOR, 1))
      t = math.ceil(self.view_start / minor) * minor
      while t <= self.view_end:
        tx = int(self.time_to_x(t))
        p.drawLine(tx, ruler_y, tx, ruler_y + 3)
        t += minor
      p.setPen(QPen(theme.RULER, 1))
      t = math.ceil(self.view_start / major) * major
      while t <= self.view_end:
        tx = int(self.time_to_x(t))
        p.drawLine(tx, ruler_y, tx, ruler_y + 6)
        p.drawText(tx + 2, ruler_y + 15, fmt_tick(t, major))
        t += major
      # region boundary handles, drawn in the ruler band only
      p.setPen(QPen(theme.REGION_MARKER, 3))
      for r in self.regions:
        for t in (r.start, r.end):
          rx = int(self.time_to_x(t))
          if 0 <= rx < w:
            p.drawLine(rx, ruler_y, rx, h)

    p.end()
