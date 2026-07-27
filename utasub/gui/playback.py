"""Playback for MainWindow: media player setup, playhead sync, plus the
playback toolbar (play button, region bounds, SRT offset/lead).
Space / T live as menu actions in actions.py."""
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWidgets import (
  QToolBar, QPushButton, QLineEdit, QLabel, QDoubleSpinBox,
)

from ..core import export as export_mod
from . import theme
from .util import settings

try:
  from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
except ImportError:  # QtMultimedia optional: playback degrades gracefully
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

    play_tb.addWidget(QLabel("Start:"))
    self._region_start_edit = QLineEdit()
    self._region_start_edit.setMaximumWidth(80)
    self._region_start_edit.setPlaceholderText("M:SS.ff")
    play_tb.addWidget(self._region_start_edit)

    play_tb.addWidget(QLabel("End:"))
    self._region_end_edit = QLineEdit()
    self._region_end_edit.setMaximumWidth(80)
    self._region_end_edit.setPlaceholderText("M:SS.ff")
    play_tb.addWidget(self._region_end_edit)

    self._apply_bounds_btn = QPushButton("Apply bounds")
    self._apply_bounds_btn.setMaximumWidth(100)
    self._apply_bounds_btn.clicked.connect(self._on_apply_bounds)
    play_tb.addWidget(self._apply_bounds_btn)

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
      self._play_timer = QTimer()
      self._play_timer.setInterval(30)
      self._play_timer.timeout.connect(self._update_playhead)
      self._player.playbackStateChanged.connect(self._on_playback_state)
      self._playback_ok = True
    except Exception as e:
      print(f"  playback unavailable ({e})")
      self._player = None

  def _on_playback_state(self, state):
    """Reset the button when playback stops on its own (end of media)."""
    playing = state == QMediaPlayer.PlayingState
    self._set_play_icon(playing)
    if not playing and self._play_timer is not None:
      self._play_timer.stop()

  def _update_playhead(self):
    if self._player is None:
      return
    t = self._player.position() / 1000.0
    self._timeline.canvas.set_playhead(t)
    self._timeline.canvas.update()

  def _on_seek(self, t):
    if self._player and self._playback_ok:
      self._player.setPosition(int(t * 1000))
      self._timeline.canvas.set_playhead(t)
      self._timeline.canvas.update()

  def _toggle_play(self):
    if not self._player or not self._playback_ok:
      return
    if self._player.playbackState() == QMediaPlayer.PlayingState:
      self._player.pause()
      self._play_timer.stop()
      self._set_play_icon(False)
    else:
      self._player.play()
      self._play_timer.start()
      self._set_play_icon(True)

  def _on_play_region(self):
    """Play/stop from selected region start."""
    if not self._player or not self._playback_ok:
      return
    if self._player.playbackState() == QMediaPlayer.PlayingState:
      self._player.pause()
      self._play_timer.stop()
      self._set_play_icon(False)
      return
    if self._active_region is not None and self._active_region < len(self._regions):
      region = self._regions[self._active_region]
      self._player.setPosition(int(region.start * 1000))
      self._timeline.canvas.playhead = region.start
    self._player.play()
    self._play_timer.start()
    self._set_play_icon(True)
