"""Background pipeline jobs for MainWindow: prepare, align, finalize, embed.
One idiom throughout: a QThread subclass whose _work() returns the payload."""
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

class _Job(QThread):
  """Runs _work(*args) off the main thread. Result goes out on `done`, any
  exception on `error`. QThread's own finished() still marks thread exit, so a
  caller can release the reference only once the thread is really gone."""
  done = Signal(object)
  error = Signal(str)

  def __init__(self, *args, **kwargs):
    super().__init__()
    self._args, self._kwargs = args, kwargs

  def run(self):
    try:
      self.done.emit(self._work(*self._args, **self._kwargs))
    except Exception as e:
      traceback.print_exc()
      self.error.emit(str(e))

class _FinalizeWorker(_Job):
  """Align every region for editing; export is a separate, explicit step."""

  def _work(self, *args, **kwargs):
    from ..core.multi_song import _multi_song_finalize
    return _multi_song_finalize(*args, export=False, **kwargs)

class _AlignRegionWorker(_Job):
  """Align one region; the index rides along so a late result stays attributable."""

  def _work(self, segments, audio, region, region_idx, assignment, opts,
            regions=None, scored=None):
    from ..core.multi_song import _finalize_single_region
    cues, meta = _finalize_single_region(segments, audio, region, region_idx,
                                         assignment, opts, regions, scored=scored)
    return cues, meta, region_idx

class _SetlistAssignWorker(_Job):
  """Setlist DP over the regions for a pasted setlist (fetch + score)."""

  def _work(self, *args):
    from ..core.multi_song import _setlist_dp_assign
    return _setlist_dp_assign(*args, notes=[], source_label="Setlist:Manual")

class _EmbedWorker(_Job):
  """ffmpeg mux of the exported subtitle sidecars into the video."""

  def _work(self, media_path, sub_paths):
    from ..core.export import embed_subs
    return str(embed_subs(media_path, sub_paths))

class _TranslateWorker(_Job):
  """LLM translation of cue texts over the OpenAI-compatible endpoint. Off-thread
  because a batch on a local model can take a minute."""

  def _work(self, texts, target, endpoint, model):
    from ..core.translate import translate_lines
    return translate_lines(texts, target, endpoint, model=model)

class _PrepareWorker(_Job):
  """ASR + audio load, then region prepare. session_only=True stops after the
  media load: a saved session already holds its cues and regions, only the
  waveform, transcript and tags are missing, and all three would block the GUI
  thread for seconds if loaded inline."""

  def _work(self, path, providers=None, fresh=False, use_setlist=True,
            session_only=False):
    from ..core.asr import load_asr
    from ..core.providers import media_tags
    from ..core.multi_song import _load_audio_once, _multi_song_prepare
    from .timeline_util import find_onsets

    path = Path(path)
    print(f"=== {path.name}")
    result = load_asr(path)
    if result is None and not session_only:
      raise RuntimeError(
        f"No transcript for {path.name}. Use the Transcribe button (needs the "
        f"[asr] extra), or run utasub-asr on the file first.")
    segments = result[0] if result else []
    print(f"  loaded {len(segments)} ASR segments")

    audio, audio_path = _load_audio_once(path)
    if audio is not None and str(audio_path) != str(path):
      print(f"  audio: {Path(audio_path).name}")
    tags = media_tags(path)
    data = {"path": str(path), "tags": tags, "segments": segments,
            "media_dur_ms": tags.get("_duration_ms", 0), "audio": audio,
            "onsets": [], "regions": [], "region_scored": [], "assignments": []}
    if session_only:
      data["onsets"] = find_onsets(audio) if audio is not None else []
      return data

    prep = _multi_song_prepare(path, segments, audio, providers or [], fresh,
                               use_setlist=use_setlist)
    if prep is not None:  # no regions detected: load with empty regions
      regions, per_region_scored, assignments, _notes, _mc_flags = prep
      data.update(regions=regions, region_scored=per_region_scored,
                  assignments=assignments)
    return data
