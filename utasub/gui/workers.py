"""Background pipeline workers for MainWindow: prepare, align, finalize, embed."""
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal


class _FinalizeWorker(QObject):
  """Runs _multi_song_finalize off the main thread (align only, no export)."""
  finished = Signal(object)  # (all_cues, region_metas) or None on error
  error = Signal(str)

  def __init__(self, path, segments, audio, regions, assignments, romaji, opts):
    super().__init__()
    self._args = (path, segments, audio, regions, assignments, romaji, opts)

  def run(self):
    try:
      from ..cli import _multi_song_finalize
      self.finished.emit(_multi_song_finalize(*self._args, export=False))
    except Exception as e:
      traceback.print_exc()
      self.error.emit(str(e))


class _AlignRegionWorker(QObject):
  """Runs _finalize_single_region off the main thread for 'Align region'."""
  finished = Signal(object)  # (cues, meta, region_idx) or None on error
  error = Signal(str)

  def __init__(self, segments, audio, region, region_idx, assignment, opts,
               regions=None):
    super().__init__()
    self._args = (segments, audio, region, region_idx, assignment, opts, regions)
    self._region_idx = region_idx

  def run(self):
    try:
      from ..cli import _finalize_single_region
      cues, meta = _finalize_single_region(*self._args)
      self.finished.emit((cues, meta, self._region_idx))
    except Exception as e:
      traceback.print_exc()
      self.error.emit(str(e))


class _SetlistAssignWorker(QObject):
  """Runs the setlist DP for a pasted setlist off the main thread (fetch + score)."""
  finished = Signal(object)  # (assignments, per_region_scored) or None on error
  error = Signal(str)

  def __init__(self, regions, tracklist, segments, providers, artist_hint, fresh):
    super().__init__()
    self._args = (regions, tracklist, segments, providers, artist_hint, fresh)

  def run(self):
    try:
      from ..cli import _setlist_dp_assign
      self.finished.emit(_setlist_dp_assign(*self._args, notes=[],
                                            source_label="Setlist:Manual"))
    except Exception as e:
      traceback.print_exc()
      self.error.emit(str(e))


class _PrepareWorker(QThread):
  """Background: load ASR + audio + run _multi_song_prepare."""
  finished_data = Signal(object)  # dict with all loaded data, or None
  error = Signal(str)

  def __init__(self, path, providers, fresh, use_setlist):
    super().__init__()
    self._path = path
    self._providers = providers
    self._fresh = fresh
    self._use_setlist = use_setlist

  def run(self):
    try:
      from ..core.asr import load_asr
      from ..core.providers import media_tags
      from ..cli import _load_audio_once, _multi_song_prepare

      path = Path(self._path)
      print(f"=== {path.name}")

      result = load_asr(path)
      if result is None:
        self.error.emit(
          f"No transcript for {path.name}. "
          f"Use the Transcribe button (needs the [asr] extra), or run "
          f"utasub-asr on the file first.")
        return
      segments, language = result
      print(f"  loaded {len(segments)} ASR segments")

      audio, audio_path = _load_audio_once(path)
      if audio is not None and str(audio_path) != str(path):
        print(f"  audio: {Path(audio_path).name}")

      tags = media_tags(path)
      media_dur_ms = tags.get("_duration_ms", 0)

      prep = _multi_song_prepare(
        path, segments, audio, self._providers, self._fresh,
        use_setlist=self._use_setlist)

      if prep is None:  # no regions detected: still load with empty regions
        self.finished_data.emit({
          "path": str(path), "tags": tags, "segments": segments,
          "media_dur_ms": media_dur_ms, "audio": audio,
          "regions": [], "region_scored": [], "assignments": [],
        })
        return

      regions, per_region_scored, assignments, notes = prep
      self.finished_data.emit({
        "path": str(path), "tags": tags, "segments": segments,
        "media_dur_ms": media_dur_ms, "audio": audio,
        "regions": regions, "region_scored": per_region_scored,
        "assignments": assignments,
      })
    except Exception as e:
      traceback.print_exc()
      self.error.emit(str(e))


class _EmbedWorker(QThread):
  """Background ffmpeg mux of an SRT into the video."""
  done = Signal(str)   # output path, or "" on failure
  error = Signal(str)

  def __init__(self, media_path, srt_path):
    super().__init__()
    self._media_path = media_path
    self._srt_path = srt_path

  def run(self):
    try:
      from ..core.export import embed_srt
      out = embed_srt(self._media_path, self._srt_path)
      self.done.emit(str(out))
    except Exception as e:
      traceback.print_exc()
      self.error.emit(str(e))
