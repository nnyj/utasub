"""Playback for MainWindow: media player setup, playhead sync, plus the
playback toolbar (play button, region bounds, SRT offset/lead).
Space / T live as menu actions in actions.py."""
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
  QToolBar, QPushButton, QLineEdit, QLabel, QDoubleSpinBox,
)

from ..core import export as export_mod
from . import theme
from .timeline_util import LEAD_IN_S
from .util import settings

try:
  from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
except ImportError:  # QtMultimedia optional: playback disabled if absent
  QMediaPlayer = QAudioOutput = None

class PlaybackMixin:
  """Playback toolbar + transport. Mixed into MainWindow, uses its state."""

  def _build_playback_toolbar(self):
    """Second toolbar row, visible on both tabs."""
    self.addToolBarBreak()
    play_tb = QToolBar("Playback")
    self.addToolBar(play_tb)

    self._play_region_btn = QPushButton()
    self._play_region_btn.setMaximumWidth(48)
    self._play_region_btn.clicked.connect(self._on_play_region)
    self._set_play_icon(False)
    play_tb.addWidget(self._play_region_btn)

    play_tb.addWidget(QLabel("Region start:"))
    self._region_start_edit = QLineEdit()
    self._region_end_edit = QLineEdit()
    for edit in (self._region_start_edit, self._region_end_edit):
      edit.setMaximumWidth(80)
      edit.setPlaceholderText("M:SS.ff")
      edit.setToolTip("Bounds of the region selected in the Regions dock")
    play_tb.addWidget(self._region_start_edit)
    play_tb.addWidget(QLabel("end:"))
    play_tb.addWidget(self._region_end_edit)

    self._apply_bounds_btn = QPushButton("Apply bounds")
    self._apply_bounds_btn.setMaximumWidth(110)
    self._apply_bounds_btn.clicked.connect(self._on_apply_bounds)
    play_tb.addWidget(self._apply_bounds_btn)
    self.set_region_bounds(None)

    play_tb.addSeparator()
    play_tb.addWidget(QLabel("SRT offset:"))
    self._offset_spin = self._export_spin(
      "export/offset", export_mod.SRT_OFFSET_S, (-5.0, 5.0),
      "Shifts the exported SRT; negative = subtitles appear earlier.\n"
      "Applied at write time only, so the timeline stays on the vocal onset.")
    play_tb.addWidget(self._offset_spin)

    play_tb.addWidget(QLabel("Lead:"))
    self._lead_spin = self._export_spin(
      "export/lead", export_mod.SRT_LEAD_S, (0.0, 5.0),
      "Extra head start: cues begin this much earlier, ends unchanged,\n"
      "so a line can overlap the one before it. Write time only.")
    play_tb.addWidget(self._lead_spin)

    self._view_menu.addAction(play_tb.toggleViewAction())

  @staticmethod
  def _export_spin(key, default, span, tooltip):
    """Seconds spinbox backed by a persisted export pref."""
    spin = QDoubleSpinBox()
    spin.setRange(*span)
    spin.setDecimals(2)
    spin.setSingleStep(0.05)
    spin.setValue(float(settings().value(key, default)))
    spin.valueChanged.connect(lambda v: settings().setValue(key, v))
    spin.setSuffix(" s")
    spin.setMaximumWidth(74)
    spin.setToolTip(tooltip)
    return spin

  def set_region_bounds(self, region):
    """Show the active region's bounds, or blank + disabled with no region, so
    Apply bounds is never armed with nothing to apply."""
    from .timeline_util import fmt_time_mssff
    for edit, val in ((self._region_start_edit, region and region.start),
                      (self._region_end_edit, region and region.end)):
      edit.setText("" if region is None else fmt_time_mssff(val))
      edit.setEnabled(region is not None)
    self._apply_bounds_btn.setEnabled(region is not None)

  # --- transport ---

  def _set_play_icon(self, playing):
    if playing:
      self._play_region_btn.setIcon(theme.icon("stop", theme.STOP_RED))
      self._play_region_btn.setToolTip("Stop (Space = play/pause, T = tap sync)")
    else:
      self._play_region_btn.setIcon(theme.icon("play", theme.PLAY_GREEN))
      self._play_region_btn.setToolTip("Play region (Space = play/pause, T = tap sync)")

  def _setup_playback(self):
    """QMediaPlayer shared across tabs. Graceful degrade."""
    if not self._media_path or QMediaPlayer is None:
      return
    try:
      self._player = QMediaPlayer()
      self._audio_output = QAudioOutput()
      self._player.setAudioOutput(self._audio_output)
      self._player.setSource(QUrl.fromLocalFile(str(self._media_path)))
      # positionChanged lands every ~50ms on the Qt6 backend, close enough to
      # drive the playhead without a second timer polling position()
      self._player.positionChanged.connect(self._on_position)
      self._player.playbackStateChanged.connect(self._on_playback_state)
      self._playback_ok = True
    except Exception as e:
      print(f"  playback unavailable ({e})")
      self._player = None

  def _on_playback_state(self, state):
    """Reset the button when playback stops on its own (end of media)."""
    self._set_play_icon(state == QMediaPlayer.PlayingState)

  def _on_position(self, ms):
    self._timeline.canvas.set_playhead(ms / 1000.0)  # set_playhead repaints

  def _on_seek(self, t):
    if self._player and self._playback_ok:
      self._player.setPosition(int(t * 1000))
    self._timeline.canvas.set_playhead(t)

  def _play_at(self, t=None):
    """Play from t. Without a t this is a plain toggle, with one it restarts
    there even mid-playback."""
    if not self._player or not self._playback_ok:
      return
    if t is None and self._player.playbackState() == QMediaPlayer.PlayingState:
      self._player.pause()
      self._set_play_icon(False)
      return
    if t is not None:
      self._player.setPosition(int(max(t, 0.0) * 1000))
      self._timeline.canvas.set_playhead(max(t, 0.0))
    self._player.play()
    self._set_play_icon(True)

  def _play_from_cue(self):
    """Ctrl+Space: play the selected cue with a short lead-in."""
    canvas = self._timeline.canvas
    idx = canvas.selected
    if not (0 <= idx < len(canvas.cues)):
      self._say("  play from cue: no cue selected")
      return
    self._play_at(canvas.cues[idx].start - LEAD_IN_S)

  def _on_play_region(self):
    """Play/stop from the selected region's start. Mid-playback the button is a
    stop, so a region start is only passed when starting."""
    region = (self._regions[self._active_region]
              if self._active_region is not None
              and self._active_region < len(self._regions) else None)
    playing = (self._player is not None
               and self._player.playbackState() == QMediaPlayer.PlayingState)
    self._play_at(None if playing or not region else region.start)
