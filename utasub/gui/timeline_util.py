"""Timeline primitives shared by the canvas, cue grid and panel:
romanization cache, onset detection, time format/parse, undo stack, constants."""
import re
from functools import lru_cache


@lru_cache(maxsize=None)
def romaji_for(text):
  """Cached romanization for cue text (shared by canvas + grid). Empty if no JP."""
  from ..core.romanize import is_cjk, romanize
  return romanize(text) if is_cjk(text) else ""


# --- onset detection ---

def find_onsets(audio, sr=16000, gate=0.08):
  """High-gate onset times from RMS envelope. Returns sorted list of seconds."""
  import numpy as np
  if audio is None or len(audio) < sr:
    return []
  frame = int(sr * 0.05)
  n = len(audio) // frame
  if n == 0:
    return []
  rms = np.sqrt((audio[:n * frame].reshape(n, frame)
                 .astype(np.float64) ** 2).mean(axis=1))
  threshold = max(float(rms.max()) * gate, 1e-3)
  onsets = []
  prev_voiced = False
  for i, v in enumerate(rms):
    voiced = v > threshold
    if voiced and not prev_voiced:
      onsets.append(i * 0.05)
    prev_voiced = voiced
  return onsets


# --- undo/redo command stack ---

class UndoStack:
  """Command stack: timing edits, insert/remove ops."""

  def __init__(self):
    self._stack = []
    self._pos = -1  # points to last applied command

  def push(self, cmd):
    """cmd: ('timing', idx, old_cue, new_cue[, [(idx, old, new), ...] rippled])
           | ('insert', idx, cue)
           | ('remove', idx, cue)"""
    self._stack = self._stack[:self._pos + 1]  # discard redo history
    self._stack.append(cmd)
    self._pos = len(self._stack) - 1

  def undo(self):
    """Returns cmd to undo, or None."""
    if self._pos < 0:
      return None
    cmd = self._stack[self._pos]
    self._pos -= 1
    return cmd

  def redo(self):
    """Returns cmd to redo, or None."""
    if self._pos + 1 >= len(self._stack):
      return None
    self._pos += 1
    return self._stack[self._pos]

  def clear(self):
    self._stack.clear()
    self._pos = -1


# --- drag modes ---

DRAG_NONE = 0
DRAG_MOVE = 1       # drag body = move both bounds
DRAG_LEFT = 2       # drag left edge
DRAG_RIGHT = 3      # drag right edge
DRAG_PAN = 4        # pan canvas (drag empty space)
DRAG_REGION = 5     # drag region boundary marker

EDGE_THRESH_PX = 8  # pixels from edge to trigger edge drag
NEIGHBOR_SNAP_PX = 8  # pixels from a neighbour cue's edge to snap a resize onto it
REGION_THRESH_PX = 8  # pixels from region marker to trigger drag
MIN_REGION_S = 5.0  # minimum region length in seconds
CANVAS_H = 200      # default waveform block height; splitter drag overrides


# --- time formatting ---

def fmt_time_mssff(seconds):
  """Format seconds as M:SS.ff (consistent with ruler)."""
  m, s = divmod(abs(seconds), 60)
  return f"{int(m)}:{s:05.2f}"


TICK_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]


def fmt_tick(seconds, step):
  """Ruler label: sub-second precision only if step needs it; hours shown
  only past the hour mark (keeps long-concert labels short)."""
  h, rem = divmod(abs(seconds), 3600)
  m, s = divmod(rem, 60)
  if step < 1:
    body = f"{int(m)}:{s:04.1f}"
  else:
    body = f"{int(m):02d}:{int(s):02d}" if h else f"{int(m)}:{int(s):02d}"
  return f"{int(h)}:{body}" if h else body


def parse_time(text):
  """Parse M:SS.ff or plain seconds. Returns float or None."""
  text = text.strip()
  m = re.match(r'^(\d+):(\d{1,2}(?:\.\d*)?)$', text)
  if m:
    return int(m.group(1)) * 60 + float(m.group(2))
  try:
    return float(text)
  except ValueError:
    return None
