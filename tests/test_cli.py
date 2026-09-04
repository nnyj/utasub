"""CLI-level behavior: target expansion, manual-edit survival across re-align."""
import tempfile
from pathlib import Path

from utasub.cli import expand_targets
from utasub.core import session
from utasub.core.multi_song import export_and_save

def _touch(root, names):
  for n in names:
    p = Path(root) / n
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()

def test_directory_target_expands_to_media_files_only():
  with tempfile.TemporaryDirectory() as d:
    _touch(d, ["a.mkv", "b.mp4", "c.flac", "notes.txt", "a.utasub.json",
               "sub/deep.mkv"])
    names = [p.name for p in expand_targets([d])]
    assert names == ["a.mkv", "b.mp4", "c.flac"], names

def test_recursive_target_descends():
  with tempfile.TemporaryDirectory() as d:
    _touch(d, ["a.mkv", "sub/deep.mkv", "sub/skip.txt"])
    names = sorted(p.name for p in expand_targets([d], recursive=True))
    assert names == ["a.mkv", "deep.mkv"], names

def test_literal_glob_is_expanded_and_duplicates_dropped():
  """cmd and Explorer never expand *.mkv, so the CLI does."""
  with tempfile.TemporaryDirectory() as d:
    _touch(d, ["a.mkv", "b.mkv", "c.mp4"])
    got = expand_targets([f"{d}/*.mkv", f"{d}/a.mkv"])
    assert [p.name for p in got] == ["a.mkv", "b.mkv"], got

def test_directory_target_skips_this_tools_own_outputs():
  """A directory run that ingests its own .subbed.mkv re-runs stems and ASR on
  a file already done, and the next run picks up .subbed.subbed.mkv."""
  with tempfile.TemporaryDirectory() as d:
    _touch(d, ["Neon Sky.mkv", "Neon Sky.subbed.mkv", "Neon Sky.subbed.subbed.mkv",
               "Neon Sky.vocals.flac", "Neon Sky.instrumental.flac",
               "Neon Sky.utasub.json"])
    assert [p.name for p in expand_targets([d])] == ["Neon Sky.mkv"]
    assert [p.name for p in expand_targets([f"{d}/*.mkv"])] == ["Neon Sky.mkv"]

def test_a_named_output_file_is_still_processed():
  """The skip is for expansion; asking for the file by name means it."""
  with tempfile.TemporaryDirectory() as d:
    _touch(d, ["Neon Sky.subbed.mkv"])
    got = expand_targets([f"{d}/Neon Sky.subbed.mkv"])
    assert [p.name for p in got] == ["Neon Sky.subbed.mkv"]

def test_plain_file_target_passes_through_unchecked():
  got = expand_targets(["missing.mkv"])
  assert [str(p) for p in got] == ["missing.mkv"]

def test_manual_edits_survive_a_rewrite_of_the_session():
  """A re-align rebuilds every cue; hand-corrected timings must come back and
  stay in the session file for the next run."""
  with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "concert.mkv"
    path.touch()
    edits = {"timings": [{"text": "line two", "at": 20.0,
                          "start": 18.5, "end": 22.0}], "deleted": []}
    cues = [(10.0, 14.0, "line one"), (20.0, 24.0, "line two")]
    export_and_save(path, cues, romaji=False, edits=edits)
    saved = session.load(path)
    assert saved["cues"][1][0] == 18.5, saved["cues"]
    assert saved["manual_edits"]["timings"] == edits["timings"]
