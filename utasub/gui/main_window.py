"""MainWindow: unified region-list + picker/timeline tabs + log dock.
Supports empty-state launch with Open... (Ctrl+O) to pick a file."""
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QThread, QUrl
from PySide6.QtWidgets import (
  QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel,
  QApplication, QMessageBox, QFileDialog, QInputDialog,
)

from . import theme
from .actions import build_actions
from .picker import PickerPanel, build_alternates
from .playback import PlaybackMixin
from .region_list import RegionListMixin
from .timeline import TimelinePanel
from .log import LogDock, install_tee, uninstall_tee
from .util import install_shift_hscroll
from .workers import (
  _FinalizeWorker, _AlignRegionWorker, _PrepareWorker, _EmbedWorker,
  _SetlistAssignWorker,
)

MEDIA_FILTER = "Media files (*.mkv *.mp4 *.webm *.m4a *.mp3 *.wav *.flac *.ogg *.opus *.avi);;All files (*)"


class MainWindow(RegionListMixin, PlaybackMixin, QMainWindow):
  """Unified GUI: region list (left), Picker/Timeline tabs (center), log (bottom).
  Can launch empty (no file) and load via Open... or load_prepared_data()."""

  def __init__(self, parent=None, *,
               media_path=None, tags=None, segments=None, media_dur_ms=0,
               providers=None, fresh=False, regions=None, region_scored=None,
               region_assignments=None, audio=None,
               romaji=True, opts=None, session_only=None):
    super().__init__(parent)
    from ..core.place import AlignOpts
    # config (persist across file opens)
    self._providers = providers or ["NetEase"]
    self._fresh = fresh
    self._romaji = romaji
    self._opts = opts or AlignOpts()  # forced alignment toggle

    # per-file state (reset on each load)
    self._media_path = None
    self._tags = {}
    self._segments = []
    self._media_dur_ms = 0
    self._regions = []
    self._audio = None
    self._region_scored = {}
    self._region_chosen = {}
    self._region_metas = []
    self._manual_edits = {"timings": [], "deleted": []}
    self._active_region = None
    self._exported = False

    # workers
    self._worker_thread = None
    self._worker = None
    self._prepare_worker = None  # _PrepareWorker (QThread subclass)
    self._embed_worker = None
    # ASR runs out-of-process (QProcess): a torch/CUDA/native crash kills only the
    # child, never this GUI. Its output is streamed to the log; a tail is kept for
    # the failure dialog.
    self._transcribe_proc = None
    self._transcribe_tail = []
    self._pending_transcribe = None  # media path awaiting reopen after ASR

    # playback
    self._player = None
    self._play_timer = None
    self._playback_ok = False

    self.setWindowTitle("utasub")
    self.resize(1200, 700)

    self._log_dock = LogDock("Log", self)
    self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)
    install_tee(self._log_dock.bridge)

    self._build_region_dock()

    # --- center: tab widget ---
    self._tabs = QTabWidget()
    self.setCentralWidget(self._tabs)

    # picker tab: wrap panel (playback controls live in the Playback toolbar)
    self._picker_container = QWidget()
    picker_layout = QVBoxLayout(self._picker_container)
    picker_layout.setContentsMargins(0, 0, 0, 0)

    self._hint_label = QLabel("Open a media file (Ctrl+O) to begin")
    self._hint_label.setAlignment(Qt.AlignCenter)
    self._hint_label.setStyleSheet("color: #888; font-size: 14px; padding: 40px;")
    picker_layout.addWidget(self._hint_label)

    self._picker = PickerPanel(self, providers=self._providers, fresh=self._fresh)
    self._picker.results_ready.connect(self._on_picker_results)
    self._picker.apply_requested.connect(self._on_apply_lyrics)
    self._picker.setVisible(False)
    picker_layout.addWidget(self._picker, 1)

    self._tabs.addTab(self._picker_container, "Picker")

    self._timeline = TimelinePanel(self)
    self._timeline.canvas.seek_requested.connect(self._on_seek)
    self._timeline.canvas.region_changed.connect(self._on_region_boundary_dragged)
    self._timeline.canvas.playing_region_changed.connect(
      self._on_playing_region_changed)
    self._tabs.addTab(self._timeline, "Timeline")

    build_actions(self)
    self._build_playback_toolbar()

    if session_only:
      self.load_session(session_only)
    elif media_path and segments:
      self.load_prepared_data(
        media_path=media_path, tags=tags or {}, segments=segments,
        media_dur_ms=media_dur_ms, audio=audio,
        regions=regions or [], region_scored=region_scored or [],
        region_assignments=region_assignments or [])

  # --- user feedback ---

  def _say(self, msg):
    """Log line, also shown in the status bar, for user-triggered outcomes
    (else buried in the log dock)."""
    print(msg)
    self.statusBar().showMessage(msg.strip(), 4000)

  # --- lyric language ---

  def _on_lang_changed(self, lang):
    from ..core.romanize import set_lang_override
    from .util import settings
    set_lang_override(lang)
    settings().setValue("lyrics/lang", lang)
    print(f"  lyric language: {lang}")
    from .timeline_util import romaji_for
    romaji_for.cache_clear()  # cached per text, not per language
    self._timeline.grid.load_cues(self._timeline.canvas.cues)
    self._timeline.canvas.update()

  # --- open file ---

  def _last_dir(self):
    """Folder the last Open/Transcribe/Cleanup dialog used (registry-backed)."""
    from .util import settings
    return settings().value("paths/last_dir", "", str)

  def _remember_dir(self, path):
    """Persist the folder of a chosen file/dir for the next dialog."""
    if not path:
      return
    from .util import settings
    d = str(Path(path) if Path(path).is_dir() else Path(path).parent)
    settings().setValue("paths/last_dir", d)

  def _on_open(self):
    """Open... dialog, then background prepare."""
    if self._media_path and self._timeline.canvas.dirty:
      reply = QMessageBox.question(
        self, "Unsaved changes",
        "Timeline has unsaved edits. Discard and open new file?",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
      if reply != QMessageBox.Yes:
        return

    path, _ = QFileDialog.getOpenFileName(self, "Open media file", self._last_dir(), MEDIA_FILTER)
    if not path:
      return
    self._remember_dir(path)
    self._open_file(path)

  def _open_file(self, path):
    """Open a media file: reuse its saved session when one exists, else prepare
    from scratch (ASR + region detect + search)."""
    self._cleanup_prepare_thread()

    from ..core.session import session_path, has_session, has_asr
    if has_session(path):
      box = QMessageBox(self)
      box.setWindowTitle("Session found")
      box.setText(f"{session_path(path).name} has saved cues for this file.")
      box.setInformativeText("Load the saved cues, or re-analyze from scratch?")
      load_btn = box.addButton("Load session", QMessageBox.AcceptRole)
      box.addButton("Re-analyze", QMessageBox.DestructiveRole)
      box.addButton(QMessageBox.Cancel)
      box.setDefaultButton(load_btn)
      box.exec()
      clicked = box.clickedButton()
      if clicked is None or box.buttonRole(clicked) == QMessageBox.RejectRole:
        return
      if clicked is load_btn:
        print(f"\n=== Loading session for {Path(path).name}")
        self.load_session(path)
        return

    # no transcript yet: offer to generate one before analysis
    if not has_asr(path):
      reply = QMessageBox.question(
        self, "No transcript",
        f"No transcript for {Path(path).name}.\n"
        "Generate it now with Qwen3-ASR? (needs the [asr] extra installed)",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
      if reply == QMessageBox.Yes:
        self._start_transcribe(path)
        return
      self._hint_label.setText(
        f"{Path(path).name} needs a transcript before analysis "
        "(File > Transcribe)")
      self._hint_label.setVisible(True)
      return

    print(f"\n=== Opening {Path(path).name}...")
    self._prepare_worker = _PrepareWorker(
      path, self._providers, self._fresh, use_setlist=True)
    self._prepare_worker.finished_data.connect(self._on_prepare_done)
    self._prepare_worker.error.connect(self._on_prepare_error)
    self._hint_label.setText(f"Loading {Path(path).name}...")
    self._prepare_worker.start()

  def _cleanup_prepare_thread(self):
    """Ensure prepare thread is fully stopped before touching shared state."""
    if self._prepare_worker is not None and self._prepare_worker.isRunning():
      self._prepare_worker.wait(10000)
    self._prepare_worker = None

  def _on_prepare_done(self, data):
    self._cleanup_prepare_thread()
    if data is None:
      return
    self.load_prepared_data(
      media_path=data["path"], tags=data["tags"],
      segments=data["segments"], media_dur_ms=data["media_dur_ms"],
      audio=data["audio"], regions=data["regions"],
      region_scored=data["region_scored"],
      region_assignments=data["assignments"])

  def _on_prepare_error(self, msg):
    self._cleanup_prepare_thread()
    print(f"  error: {msg}")
    self._hint_label.setText(f"Error: {msg}")
    self._hint_label.setVisible(True)

  # --- transcribe (ASR generation) ---

  def _on_transcribe(self):
    """Transcribe button: use the loaded file, else prompt for one."""
    path = self._media_path
    if not path:
      path, _ = QFileDialog.getOpenFileName(self, "Transcribe media file", self._last_dir(), MEDIA_FILTER)
      if not path:
        return
      self._remember_dir(path)
    self._start_transcribe(path)

  def _start_transcribe(self, path):
    """Run ASR in a child process (crash isolation, see field comments above),
    stream output to the log, reopen the file when the transcript lands."""
    if self._transcribe_proc is not None:
      return  # already running
    from ..core.asr_gen import asr_available
    if not asr_available():
      QMessageBox.warning(
        self, "ASR not available",
        "The Qwen3-ASR model package is not installed.\n"
        "Install the extra:  pip install utasub[asr]")
      return
    self._pending_transcribe = path
    self._transcribe_tail = []
    self._transcribe_act.setEnabled(False)
    self._hint_label.setText(
      f"Transcribing {Path(path).name}... (separate process, watch the log)")
    self._hint_label.setVisible(True)
    print(f"\n=== Transcribing {Path(path).name} (subprocess)")

    proc = QProcess(self)
    proc.setProcessChannelMode(QProcess.MergedChannels)
    proc.readyReadStandardOutput.connect(self._on_transcribe_output)
    proc.finished.connect(self._on_transcribe_finished)
    proc.errorOccurred.connect(self._on_transcribe_proc_error)
    self._transcribe_proc = proc
    # -u: unbuffered, so the log updates live
    proc.start(sys.executable, ["-u", "-m", "utasub.core.asr_gen", str(path)])

  def _on_transcribe_output(self):
    if self._transcribe_proc is None:
      return
    data = bytes(self._transcribe_proc.readAllStandardOutput()).decode("utf-8", "replace")
    for line in data.splitlines():
      self._log_dock.bridge.line.emit(line)  # reuse log coloring + autoscroll
      self._transcribe_tail.append(line)
    if len(self._transcribe_tail) > 40:
      del self._transcribe_tail[:-40]

  def _on_transcribe_finished(self, exit_code, exit_status):
    self._transcribe_proc = None
    self._transcribe_act.setEnabled(True)
    path = self._pending_transcribe
    crashed = exit_status == QProcess.CrashExit
    from ..core.session import has_asr
    json_ok = bool(path) and has_asr(path)
    if not crashed and exit_code == 0 and json_ok:
      self._open_file(path)  # transcript landed, proceed to analysis
      return
    if crashed:
      reason = "crashed (often out of GPU memory on long files, try --small or --cpu)"
    elif exit_code != 0:
      reason = f"exited with code {exit_code}"
    else:
      reason = "produced no transcript"
    print(f"  transcription {reason}")
    self._hint_label.setText(f"Transcription {reason} - see log")
    self._hint_label.setVisible(True)
    tail = "\n".join(self._transcribe_tail[-15:]) or "(no output captured)"
    QMessageBox.warning(self, "Transcription failed",
                        f"ASR {reason}.\n\nLast output:\n{tail}")

  def _on_transcribe_proc_error(self, err):
    # only FailedToStart needs handling here; finished() covers crash/normal exit
    if err == QProcess.FailedToStart:
      self._transcribe_proc = None
      self._transcribe_act.setEnabled(True)
      QMessageBox.warning(self, "Transcription failed",
                          f"Could not start the ASR process:\n{sys.executable} -m utasub.core.asr_gen")

  # --- embed SRT ---

  def _find_exported_srt(self):
    """Exported subtitle on disk, .ja.srt preferred over .srt. None if absent."""
    if not self._media_path:
      return None
    p = Path(self._media_path)
    for suffix in (".ja.srt", ".srt"):
      cand = p.with_suffix(suffix)
      if cand.exists():
        return cand
    return None

  def _refresh_embed_action(self):
    self._embed_act.setEnabled(self._find_exported_srt() is not None)

  def _on_embed(self):
    if not self._media_path:
      return
    if self._embed_worker is not None and self._embed_worker.isRunning():
      return
    srt = self._find_exported_srt()
    if srt is None:
      QMessageBox.information(self, "Embed SRT", "Export an SRT first.")
      return
    print(f"\n=== Embedding {srt.name} into video...")
    self._embed_act.setEnabled(False)
    self._embed_worker = _EmbedWorker(self._media_path, str(srt))
    self._embed_worker.done.connect(self._on_embed_done)
    self._embed_worker.error.connect(self._on_embed_error)
    self._embed_worker.start()

  def _on_embed_done(self, out_path):
    self._embed_worker = None
    self._refresh_embed_action()
    print(f"  -> {Path(out_path).name}")
    QMessageBox.information(self, "Embed SRT", f"Wrote {Path(out_path).name}")

  def _on_embed_error(self, msg):
    self._embed_worker = None
    self._refresh_embed_action()
    print(f"  embed error: {msg}")
    QMessageBox.warning(self, "Embed SRT", f"ffmpeg failed:\n{msg}")

  # --- cleanup stray files ---

  def _on_cleanup(self):
    """Delete stray stem / instrumental / txt files for the loaded media, or
    scan a chosen folder when nothing is loaded. Never touches the session file or outputs."""
    from ..core.align import stem_cache_dir
    targets = []
    if self._media_path:
      p = Path(self._media_path)
      for f in p.parent.iterdir():
        if not f.is_file() or f == p or not f.stem.startswith(p.stem):
          continue
        low = f.name.lower()
        if ".vocals." in low or ".instrumental." in low or ".stem." in low or low.endswith(".txt"):
          targets.append(f)
      want = f"{p.stem}.vocals".lower()
      targets += [f for f in stem_cache_dir().iterdir()
                  if f.is_file() and f.stem.lower() == want]
    else:
      folder = QFileDialog.getExistingDirectory(self, "Folder to clean", self._last_dir())
      if not folder:
        return
      self._remember_dir(folder)
      for f in Path(folder).iterdir():
        low = f.name.lower()
        if f.is_file() and (".vocals." in low or ".instrumental." in low or ".stem." in low):
          targets.append(f)

    if not targets:
      QMessageBox.information(self, "Cleanup", "No stray files found.")
      return
    listing = "\n".join(f.name for f in targets[:40])
    if len(targets) > 40:
      listing += f"\n... (+{len(targets) - 40} more)"
    reply = QMessageBox.question(
      self, "Delete stray files", f"Delete {len(targets)} file(s)?\n\n{listing}",
      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    if reply != QMessageBox.Yes:
      return
    removed = 0
    for f in targets:
      try:
        f.unlink()
        removed += 1
        print(f"  removed {f.name}")
      except Exception as e:
        print(f"  skip {f.name}: {e}")
    QMessageBox.information(self, "Cleanup", f"Removed {removed} file(s).")

  # --- load prepared data (used by constructor and open) ---

  def load_prepared_data(self, *, media_path, tags, segments, media_dur_ms,
                         audio=None, regions=None, region_scored=None,
                         region_assignments=None):
    """Populate/repopulate all panels with prepared data."""
    if self._player is not None:
      self._player.setSource(QUrl())
      self._player = None
      self._playback_ok = False
    if self._play_timer is not None:
      self._play_timer.stop()

    # reset per-file state
    self._media_path = str(media_path)
    self._tags = tags or {}
    self._segments = segments or []
    self._media_dur_ms = media_dur_ms
    self._audio = audio
    self._regions = regions or []
    self._region_scored = {}
    self._region_chosen = {}
    self._region_metas = []
    self._manual_edits = {"timings": [], "deleted": []}
    self._active_region = None
    self._exported = False

    # restore persisted strategies if a session exists and matches region count
    try:
      from ..core import session as sess_mod
      sess = sess_mod.load(media_path)
    except Exception:
      sess = None
    if sess:
      metas = sess.get("region_metas", [])
      if metas and len(metas) == len(self._regions):
        self._region_metas = metas
      self._manual_edits = sess.get("manual_edits") or self._manual_edits

    if region_scored:
      for i, scored_list in enumerate(region_scored):
        self._region_scored[i] = scored_list
    if region_assignments:
      for i, assignment in enumerate(region_assignments):
        if assignment is not None:
          self._region_chosen[i] = assignment
    if sess:  # saved picks win over this run's auto-assignments
      self._region_chosen.update(self._restore_region_choices(sess))

    self.setWindowTitle(f"{Path(media_path).name} - utasub")

    self._hint_label.setVisible(False)
    self._picker.setVisible(True)

    self._picker.media_path = str(media_path)
    self._picker.segments = segments
    self._picker.media_duration_ms = media_dur_ms
    self._prefill_picker_combos(media_path, tags)

    self._region_list.clear()
    self._populate_region_list()

    self._timeline._media_path = str(media_path)
    self._sync_timeline_session()
    if audio is not None:
      self._timeline.canvas.set_audio(audio)
    else:
      self._timeline.canvas.onsets = []
      if segments:
        self._timeline.canvas.duration = segments[-1][1] + 2.0

    self._timeline.canvas.regions = list(self._regions)
    self._timeline.canvas.cues = []
    self._timeline.canvas.undo_stack.clear()
    self._timeline.canvas.clear_dirty()
    self._timeline.canvas.fit_all()
    self._timeline.grid.load_cues([])

    self._refresh_region_actions()
    self._export_act.setEnabled(False)
    self._export_ass_act.setEnabled(False)
    self._refresh_embed_action()

    self._setup_playback()

    if self._regions:
      self._select_region_row(0)

    # provisional LRC cues from any setlist auto-assignments
    self._rebuild_provisional_cues()

  def _prefill_picker_combos(self, media_path, tags=None):
    """Fill Title/Album/Artist combos from container tags + filename alternates."""
    alts = build_alternates(str(media_path), tags)
    for combo, key in [(self._picker.title_combo, "title"),
                       (self._picker.album_combo, "album"),
                       (self._picker.artist_combo, "artist")]:
      combo.clear()
      for val, hint in alts[key]:
        label = f"{val}  ({hint})" if hint else val
        combo.addItem(label, val)
      if alts[key]:
        combo.setCurrentIndex(0)
        combo.lineEdit().setText(alts[key][0][0])

  def _restore_region_choices(self, sess):
    """Saved per-region picks, dropped when they point past the region list."""
    choices = {i: ch for i, ch in (sess.get("region_choices") or {}).items()
               if i < len(self._regions)}
    if choices:
      print(f"  restored {len(choices)} region assignment(s)")
    return choices

  # --- direct session open (--timeline flag) ---

  def load_session(self, media_path):
    """Open an existing session straight into the timeline tab for review/edit.
    No ASR needed: cues + regions come from the saved session. Returns bool."""
    from ..core import session as sess_mod
    from ..core.regions import Region
    sess = sess_mod.load(media_path)
    if sess is None:
      print(f"  timeline: no session for {Path(media_path).name}")
      self._hint_label.setText(f"No session for {Path(media_path).name}")
      return False

    # audio for waveform (best-effort; timeline still works without it)
    audio = None
    try:
      from ..core.align import load_audio_16k, find_vocals
      audio_path = find_vocals(media_path) or media_path
      audio = load_audio_16k(str(audio_path))
    except Exception as e:
      print(f"  timeline: audio load failed ({e}), no waveform")

    self._media_path = str(media_path)
    # pin locale once from all saved cues: timeline romanizes per line
    from ..core.romanize import detect, set_default_locale
    set_default_locale(detect("\n".join(c[2] for c in sess.get("cues", []) if len(c) > 2)))
    self._romaji = sess.get("romaji", True)
    if sess.get("offset") is not None:
      self._offset_spin.setValue(sess["offset"])
    if sess.get("lead") is not None:
      self._lead_spin.setValue(sess["lead"])
    self._regions = [Region.from_dict(r) for r in sess.get("regions", [])]
    self._region_metas = sess.get("region_metas", [])
    self._manual_edits = sess.get("manual_edits") or self._manual_edits
    self._audio = audio
    # ASR block feeds re-align (anchor-warp + coarse need segments, not just cues)
    from ..core.asr import load_asr
    asr = load_asr(media_path)
    self._segments = asr[0] if asr else []
    self._region_scored = {}
    self._region_chosen = self._restore_region_choices(sess)
    self._active_region = None
    self._exported = True  # session already on disk; edits re-export via Save

    self.setWindowTitle(f"{Path(media_path).name} - utasub")
    self._hint_label.setVisible(False)
    self._picker.setVisible(True)
    self._picker.media_path = str(media_path)
    self._picker.segments = self._segments
    try:
      from ..core.providers import media_tags
      tags = media_tags(media_path)
    except Exception:
      tags = None
    self._prefill_picker_combos(media_path, tags)

    self._timeline._media_path = str(media_path)
    self._timeline._session = sess
    if audio is not None:
      self._timeline.canvas.set_audio(audio)
    self._timeline.load_cues(sess.get("cues", []), regions=list(self._regions))

    self._region_list.clear()
    self._populate_region_list()
    self._setup_playback()
    self._tabs.setCurrentWidget(self._timeline)
    self._export_act.setEnabled(True)
    self._export_ass_act.setEnabled(True)
    self._refresh_embed_action()
    return True

  # --- picker glue ---

  def _store_current_choice(self):
    if self._active_region is None:
      return
    choice = self._picker.chosen_candidate()
    if choice is not None:
      self._region_chosen[self._active_region] = choice
    self._rebuild_provisional_cues()
    self._refresh_region_actions()

  def _on_picker_results(self, scored):
    if self._active_region is not None:
      self._region_scored[self._active_region] = scored
      self._store_current_choice()  # auto-selected top candidate
      self._populate_region_list()
      self._rebuild_provisional_cues()

  def _mark_user_picked(self, region_idx):
    """Flag the region's chosen candidate as a human pick, so alignment trusts it
    over the auto-confidence guards."""
    chosen = self._region_chosen.get(region_idx)
    if chosen is not None:
      chosen[1].user_picked = True

  def _clear_cues_for_apply(self):
    """Clear on-screen cues plus their region metas so an Apply starts from empty.
    Clearing the export flag re-arms the provisional preview."""
    self._region_metas = []
    self._exported = False
    self._timeline.load_cues([], regions=list(self._regions))

  def _confirm_replace_cues(self):
    """Ask before an apply throws away aligned cues. True = go ahead."""
    canvas = self._timeline.canvas
    if not canvas.cues or canvas.provisional:
      return True
    reply = QMessageBox.question(
      self, "Replace cues", f"Replace {len(canvas.cues)} aligned cues?",
      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    return reply == QMessageBox.Yes

  def _on_apply_lyrics(self):
    """Apply button: assign the picker's selected lyrics (fetched row or pasted
    text) to the active region, replacing the cues on the timeline."""
    if self._active_region is None:
      self._say("  apply lyrics: no region selected")
      return
    if self._picker.chosen_candidate() is None:
      self._say("  apply lyrics: nothing selected in the picker")
      return
    if not self._confirm_replace_cues():
      return
    pasted = self._picker.is_paste_row()
    self._clear_cues_for_apply()
    self._store_current_choice()
    self._mark_user_picked(self._active_region)
    self._populate_region_list()
    kind = "pasted lyrics" if pasted else "lyrics"
    self._say(f"  region {self._active_region + 1}: {kind} applied")

  def _lrc_region_cues(self, confidence):
    """Cues straight from each region's chosen LRC, no alignment: start = region
    start + LRC stamp, end = next line's start (last line held 4s), clamped to
    the region end. Returns [(region_idx, candidate, cues), ...]."""
    from ..core.align import Cue
    from ..core.providers import parse_lrc
    per_region = []
    for i, region in enumerate(self._regions):
      chosen = self._region_chosen.get(i)
      if chosen is None:
        continue
      lrc = getattr(chosen[1], "lrc", "")
      if not lrc:
        continue
      timed = [(t, txt) for t, txt in parse_lrc(lrc) if t is not None]
      cues = []
      for j, (t, txt) in enumerate(timed):
        start = region.start + t
        if j + 1 < len(timed):
          end = region.start + timed[j + 1][0]
        else:
          end = min(start + 4.0, region.end)
        if end <= start:
          end = min(start + 0.5, region.end)
        cues.append(Cue(start, end, txt, confidence))
      if cues:
        per_region.append((i, chosen[1], cues))
    return per_region

  def _rebuild_provisional_cues(self):
    """Dimmed preview cues from chosen candidates' LRC, until a real export.
    Cheap to rebuild; skipped once self._exported."""
    if self._exported or not self._regions:
      return
    if self._timeline.canvas.cues and not self._timeline.canvas.provisional:
      return  # real cues on the timeline: a preview rebuild must not eat them
    cues = [c for _, _, rcues in self._lrc_region_cues("provisional") for c in rcues]
    cues.sort(key=lambda c: c.start)
    self._timeline.show_provisional_cues(cues)

  # --- align all + export ---

  def _collect_assignments(self):
    self._store_current_choice()
    assignments = [None] * len(self._regions)
    for i in range(len(self._regions)):
      assignments[i] = self._region_chosen.get(i)
    return assignments

  def _session_meta(self):
    """Session metadata shared by the two sync paths. Cues are handled
    separately by save(). `placement` is what external tooling reads;
    `region_metas` is what load_prepared_data reads."""
    return {
      "regions": [r.to_dict() for r in self._regions],
      "region_metas": self._region_metas,
      "romaji": self._romaji,
      "offset": self._offset_spin.value(),
      "lead": self._lead_spin.value(),
      "region_choices": dict(self._region_chosen),
      "placement": {"multi_song": True, "regions": self._region_metas},
      "manual_edits": self._manual_edits,
    }

  def _sync_timeline_session(self):
    """Populate the embedded TimelinePanel's _session so its save() writes a
    complete, reloadable multi-song session."""
    self._timeline._session = self._session_meta()

  def _on_reset_edits(self):
    """Drop all manual timings/deletions after confirmation. Cues on screen keep
    their positions until the next Align re-places them."""
    n = len((self._manual_edits or {}).get("timings") or [])
    reply = QMessageBox.question(
      self, "Reset manual edits",
      f"Forget {n} manual timing record(s) and all deletions?\n"
      "The next Align will place every line fresh.",
      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    if reply != QMessageBox.Yes:
      return
    if not self._timeline.reset_manual_edits():
      print("  reset edits: no session loaded")
      return
    self._manual_edits = {"timings": [], "deleted": []}
    print("  reset edits: cleared (re-run Align, then Save)")

  def _pull_manual_edits(self):
    """Take over edits made since the last sync, saved or not: the cues are about
    to be rebuilt away and re-applied from here by load_cues."""
    self._manual_edits = ((self._timeline._session or {}).get("manual_edits")
                          or self._manual_edits)

  def _on_use_lrc_as_is(self):
    """Load the LRC timestamps straight into the timeline, skipping alignment, so
    trusted LRC only needs the Cue offset buttons. Replaces the cues on screen
    like Align all does; no SRT write. Synchronous (parse only)."""
    if self._worker_thread is not None or not self._media_path:
      return
    self._store_current_choice()
    per_region = self._lrc_region_cues("lrc")
    if not per_region:
      self._say("  LRC as-is: no assigned region has timestamped lyrics")
      return
    self._pull_manual_edits()
    cues = [c for _, _, rcues in per_region for c in rcues]
    cues.sort(key=lambda c: c.start)
    self._region_metas = [
      {"region_idx": i, "strategy": "lrc_as_is", "title": cand.title,
       "song_span": [round(rcues[0].start, 3), round(rcues[-1].end, 3)]}
      for i, cand, rcues in per_region]
    print(f"\n=== LRC as-is: {len(cues)} cues loaded from "
          f"{len(per_region)}/{len(self._regions)} region(s) (not exported yet)")
    self._sync_timeline_session()
    self._populate_region_list()
    self._timeline.load_cues(cues, regions=list(self._regions))
    self._tabs.setCurrentWidget(self._timeline)
    self._export_act.setEnabled(True)
    self._export_ass_act.setEnabled(True)

  def _start_worker(self, worker, done_slot):
    """Run an align worker on its own QThread. The refs are nulled only after the
    thread's event loop exits: dropping them in the result slot lets GC destroy a
    running QThread (qFatal abort)."""
    self._worker = worker
    self._worker_thread = QThread()
    worker.moveToThread(self._worker_thread)
    self._worker_thread.started.connect(worker.run)
    worker.finished.connect(done_slot)
    worker.error.connect(self._on_finalize_error)
    worker.finished.connect(self._worker_thread.quit)
    worker.error.connect(self._worker_thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.error.connect(worker.deleteLater)
    self._worker_thread.finished.connect(self._on_finalize_thread_done)
    self._worker_thread.start()

  def _on_align_all(self):
    """Align all regions into the timeline for review. No SRT write (Export SRT
    does that). Runs finalize off the main thread."""
    if self._worker_thread is not None or not self._media_path:
      return
    assignments = self._collect_assignments()
    assigned = sum(1 for a in assignments if a is not None)
    self._pull_manual_edits()
    print(f"\n=== Align all: {assigned}/{len(self._regions)} regions assigned")

    self._align_all_act.setEnabled(False)
    self._align_region_act.setEnabled(False)
    self._lrc_as_is_act.setEnabled(False)

    self._start_worker(
      _FinalizeWorker(self._media_path, self._segments, self._audio,
                      self._regions, assignments, self._romaji, self._opts),
      self._on_finalize_done)

  def _on_finalize_thread_done(self):
    if self._worker_thread is not None:
      self._worker_thread.deleteLater()
    self._worker_thread = None
    self._worker = None
    self._refresh_region_actions()

  def _on_finalize_done(self, result):
    self._align_all_act.setEnabled(True)
    if result:
      all_cues, region_metas = result
      print(f"\n=== Align complete: {len(all_cues)} cues (not exported yet)")
      self._region_metas = region_metas or []
      self._sync_timeline_session()
      self._populate_region_list()
      self._timeline.load_cues(all_cues, regions=list(self._regions))
      self._tabs.setCurrentWidget(self._timeline)
      self._export_act.setEnabled(True)
      self._export_ass_act.setEnabled(True)
    else:
      print("\n=== Align finished (no result)")

  def _on_finalize_error(self, msg):
    self._align_all_act.setEnabled(True)
    self._refresh_region_actions()
    print(f"\n=== Align error: {msg}")

  # --- pasted setlist ---

  def _on_paste_setlist(self):
    """Paste a setlist by hand, then re-run the setlist DP over the regions.
    Results feed in exactly like discovery's do (region_chosen + region_scored);
    regions the human already picked keep their pick."""
    if self._worker_thread is not None or not self._media_path or not self._regions:
      return
    text, ok = QInputDialog.getMultiLineText(
      self, "Paste setlist",
      "One title per line (numbered, bulleted or comma-separated also work):", "")
    if not ok:
      return
    from ..core.setlist import parse_pasted
    titles = parse_pasted(text)
    if not titles:
      print("  paste setlist: no titles parsed")
      return
    tracklist = [(i + 1, t) for i, t in enumerate(titles)]
    print(f"\n=== Paste setlist: {len(tracklist)} tracks, fetching + matching")
    self._align_all_act.setEnabled(False)
    self._align_region_act.setEnabled(False)
    self._lrc_as_is_act.setEnabled(False)
    self._start_worker(
      _SetlistAssignWorker(self._regions, tracklist, self._segments,
                           self._providers, self._artist_hint(), self._fresh),
      self._on_paste_setlist_done)

  def _artist_hint(self):
    """Artist text currently in the picker combo, used to qualify track queries."""
    try:
      return self._picker.artist_combo.currentText().strip()
    except Exception:
      return ""

  def _on_paste_setlist_done(self, result):
    if not result:
      return
    assignments, per_region_scored = result
    assigned = 0
    for i, pair in enumerate(assignments):
      if pair is None or getattr(self._regions[i], "mc", False):
        continue
      chosen = self._region_chosen.get(i)
      if chosen is not None and getattr(chosen[1], "user_picked", False):
        continue  # human pick wins over the DP
      self._region_chosen[i] = pair
      assigned += 1
    for i, scored in enumerate(per_region_scored):
      if scored:
        self._region_scored[i] = scored
    print(f"  paste setlist: {assigned}/{len(self._regions)} regions assigned")
    self._sync_timeline_session()
    self._populate_region_list()
    self._rebuild_provisional_cues()
    self._refresh_region_actions()

  # --- align single region ---

  def _on_align_region(self):
    """Align only the active region and merge its cues into the timeline."""
    if self._worker_thread is not None or not self._media_path:
      return
    idx = self._active_region
    if idx is None or idx >= len(self._regions):
      return
    self._store_current_choice()
    assignment = self._region_chosen.get(idx)
    if assignment is None:
      self._say("  align region: no assignment for this region")
      return
    print(f"\n=== Align region {idx+1}")
    self._align_all_act.setEnabled(False)
    self._align_region_act.setEnabled(False)
    self._lrc_as_is_act.setEnabled(False)

    self._start_worker(
      _AlignRegionWorker(self._segments, self._audio, self._regions[idx], idx,
                         assignment, self._opts, self._regions),
      self._on_align_region_done)

  def _on_align_region_done(self, result):
    if not result:
      return
    cues, meta, region_idx = result
    if region_idx >= len(self._regions):
      return
    region = self._regions[region_idx]
    kept = [c for c in self._timeline.canvas.cues
            if not (region.start - 0.5 <= c[0] and c[1] <= region.end + 0.5)]
    merged = kept + list(cues)
    merged.sort(key=lambda c: c[0])
    self._region_metas = [m for m in self._region_metas
                          if m.get("region_idx") != region_idx]
    self._region_metas.append(meta)
    self._region_metas.sort(key=lambda m: m.get("region_idx", 0))
    print(f"  region {region_idx+1}: {len(cues)} cues merged ({meta.get('strategy')})")
    self._sync_timeline_session()
    self._populate_region_list()
    self._timeline.load_cues(merged, regions=list(self._regions))
    self._tabs.setCurrentWidget(self._timeline)
    self._export_act.setEnabled(True)
    self._export_ass_act.setEnabled(True)

  # --- save / export ---

  def _on_save(self):
    """Save the session only. SRT writing belongs to Export."""
    self._refresh_timeline_session_meta()
    if self._timeline.save(export=False):
      print("  saved session")

  def _on_export(self):
    """Write SRT from current timeline cues + save session (Export SRT action)."""
    self._refresh_timeline_session_meta()
    if self._timeline.save(export=True):
      self._exported = True
      self._refresh_embed_action()
      print("  exported SRT + saved session")

  def _on_export_ass(self):
    """Write styled .ass from current timeline cues (romaji over dimmed original).
    Persists the session first so edits are not lost, but does not touch the SRT."""
    self._refresh_timeline_session_meta()
    if not self._timeline.save(export=False):
      return
    from ..core.export import export_ass
    cues = self._timeline._session.get("cues", [])
    export_ass(self._media_path, cues,
               offset=self._offset_spin.value(), lead=self._lead_spin.value())
    self._exported = True
    print("  exported ASS")

  def _refresh_timeline_session_meta(self):
    """Refresh session metadata (regions may have moved) without dropping the
    cue-collection logic in TimelinePanel.save."""
    self._timeline._session.update(self._session_meta())

  def _refresh_region_actions(self):
    """Enable Align region when the active region has an assignment; keep Align
    all enabled once a file is loaded; LRC as-is needs a timestamped assignment."""
    busy = self._worker_thread is not None
    self._align_all_act.setEnabled(bool(self._media_path) and not busy)
    has_assignment = (self._active_region is not None
                      and self._active_region in self._region_chosen)
    self._align_region_act.setEnabled(has_assignment and not busy)
    has_lrc = any(getattr(ch[1], "lrc", "") for ch in self._region_chosen.values())
    self._lrc_as_is_act.setEnabled(bool(self._media_path) and has_lrc and not busy)

  # --- cleanup ---

  def closeEvent(self, event):
    self._picker.shutdown()
    if self._player is not None:
      self._player.setSource(QUrl())
      self._player = None
    if self._play_timer is not None:
      self._play_timer.stop()
    if self._worker_thread is not None:
      self._worker_thread.quit()
      self._worker_thread.wait(3000)
    if self._prepare_worker is not None and self._prepare_worker.isRunning():
      self._prepare_worker.wait(3000)
    if self._transcribe_proc is not None:
      self._transcribe_proc.kill()
      self._transcribe_proc.waitForFinished(2000)
    if self._embed_worker is not None and self._embed_worker.isRunning():
      self._embed_worker.wait(3000)
    uninstall_tee()
    super().closeEvent(event)


def open_main_window(*, media_path=None, tags=None, segments=None,
                     media_dur_ms=0, providers=None, fresh=False,
                     regions=None, region_scored=None,
                     region_assignments=None, audio=None,
                     romaji=True, opts=None, session_only=None):
  """Launch MainWindow, blocking until closed. Returns True if user exported.
  media_path=None → empty state; session_only=path → direct timeline open."""
  app = QApplication.instance() or QApplication(sys.argv)
  theme.apply_theme(app)
  if sys.platform == "win32":
    # give the process its own taskbar slot + icon
    try:
      import ctypes
      ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("utasub.gui")
    except Exception:
      pass
  app.setWindowIcon(theme.app_icon())
  install_shift_hscroll(app)  # shift+wheel = horizontal scroll in every scroll area
  win = MainWindow(
    media_path=media_path, tags=tags, segments=segments,
    media_dur_ms=media_dur_ms, providers=providers, fresh=fresh,
    regions=regions, region_scored=region_scored,
    region_assignments=region_assignments, audio=audio,
    romaji=romaji, opts=opts, session_only=session_only)
  win.show()
  app.exec()
  return win._exported
