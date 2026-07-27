"""Session save/load round-trip, deterministic re-export."""
import tempfile
import shutil
from pathlib import Path

from utasub.core.providers import Candidate
from utasub.core import session


def _tmp_media():
  d = Path(tempfile.mkdtemp(prefix="utasub_test_"))
  media = d / "test_song.mkv"
  media.write_bytes(b"\x00")
  return media


def _sample_candidates():
  return [
    Candidate(title="Song A", album="Album", artist="Artist",
              source="NetEase", duration_ms=180000, lrc="[00:01.00] line one"),
    Candidate(title="Song B", album="", artist="Other",
              source="LRCLib", duration_ms=200000, lrc=""),
  ]


def _sample_cues():
  return [(0.0, 2.5, "hello world"), (2.5, 5.0, "second line"), (5.0, 8.0, "third line")]


def test_round_trip_preserves_all_fields():
  media = _tmp_media()
  try:
    cands = _sample_candidates()
    cues = _sample_cues()
    session.save(media, candidates=cands, chosen_index=0,
                 cues=cues, song_span=(0.0, 8.0), romaji=False)
    loaded = session.load(media)
    assert loaded is not None
    assert loaded["schema_version"] == session.SCHEMA_VERSION
    assert loaded["chosen_index"] == 0
    assert len(loaded["candidates"]) == 2
    assert loaded["candidates"][0].title == "Song A"
    assert loaded["candidates"][1].lrc == ""
    assert loaded["cues"] == cues
    assert loaded["song_span"] == (0.0, 8.0)
    assert loaded["romaji"] is False
    assert loaded["credit_toggles"] == {}
    assert loaded["regions"] == []
  finally:
    shutil.rmtree(media.parent)


def test_load_missing_returns_none():
  media = _tmp_media()
  try:
    assert session.load(media) is None
  finally:
    shutil.rmtree(media.parent)


def test_deterministic_reexport_japanese():
  """.ja.srt and .romaji.srt byte-identical across runs."""
  media = _tmp_media()
  try:
    cues = [(0.0, 3.0, "そらに浮かぶ月"), (3.0, 6.0, "風の声が聞こえる")]
    session.save(media, cues=cues, romaji=True)
    sess = session.load(media)

    bytes1 = [p.read_bytes() for p in session.reexport(media, sess)]
    bytes2 = [p.read_bytes() for p in session.reexport(media, sess)]
    assert len(bytes1) == len(bytes2) == 2
    for b1, b2 in zip(bytes1, bytes2):
      assert b1 == b2
  finally:
    shutil.rmtree(media.parent)


# --- manual edits across a re-align ---

def _c(s, e, t):
  from utasub.core.align import Cue
  return Cue(s, e, t, "lrc")


def test_edits_reattach_after_reorder_insert_delete():
  """Edits reattach by text + position made at, not by index."""
  from utasub.core.session import _edit_record, apply_manual_edits
  old = [_c(10.0, 12.0, "alpha"), _c(20.0, 22.0, "beta"), _c(30.0, 32.0, "gamma")]
  edits = {"timings": [_edit_record(old[2], start=30.5, end=33.0)], "deleted": []}
  # re-align: line inserted ahead of it, everything re-timed and renumbered
  fresh = [_c(5.0, 6.0, "intro"), _c(10.4, 12.4, "alpha"),
           _c(20.4, 22.4, "beta"), _c(30.4, 32.4, "gamma")]
  out = apply_manual_edits(fresh, edits)
  assert len(out) == 4
  assert (out[3].start, out[3].end) == (30.5, 33.0), "edit lost its line"
  assert out[1].start == 10.4, "untouched line should keep the fresh placement"


def test_edits_pick_the_right_repeat_of_a_line():
  """Repeated lyrics: edit belongs to the occurrence it was made at."""
  from utasub.core.session import _edit_record, apply_manual_edits
  old = [_c(10.0, 12.0, "chorus"), _c(50.0, 52.0, "chorus"), _c(90.0, 92.0, "chorus")]
  edits = {"timings": [_edit_record(old[1], start=50.5, end=53.0)], "deleted": []}
  fresh = [_c(10.3, 12.3, "chorus"), _c(50.3, 52.3, "chorus"), _c(90.3, 92.3, "chorus")]
  out = apply_manual_edits(fresh, edits)
  assert (out[1].start, out[1].end) == (50.5, 53.0)
  assert out[0].start == 10.3 and out[2].start == 90.3, "wrong repeat edited"


def test_deleted_cue_stays_gone_after_realign():
  """Removed cue stays gone, incl. raw ASR leftover."""
  from utasub.core.session import _edit_record, apply_manual_edits
  edits = {"timings": [],
           "deleted": [_edit_record((159.51, 164.63, "stray asr line"))]}
  fresh = [_c(159.51, 164.63, "stray asr line"), _c(160.1, 163.0, "real lyric")]
  out = apply_manual_edits(fresh, edits)
  assert len(out) == 1 and out[0].text == "real lyric"


def test_unmatched_edit_is_not_destructive():
  """Edit whose line is absent this pass leaves fresh cues alone."""
  from utasub.core.session import _edit_record, apply_manual_edits
  edits = {"timings": [_edit_record((10.0, 12.0, "absent"))], "deleted": []}
  fresh = [_c(1.0, 2.0, "present")]
  out = apply_manual_edits(fresh, edits)
  assert len(out) == 1 and (out[0].start, out[0].end) == (1.0, 2.0)


def test_old_positional_session_migrates_on_load():
  """Session predating manual_edits keeps its corrections after migrate."""
  import json
  import tempfile
  from pathlib import Path
  from utasub.core import session as sess_mod
  with tempfile.TemporaryDirectory() as tmp:
    media = Path(tmp) / "x.mkv"
    media.touch()
    raw = {
      "schema_version": 1, "media": str(media),
      "cues": [[10.0, 12.0, "alpha", "lrc"], [20.0, 22.0, "beta", "lrc"],
               [30.0, 31.0, "some asr"]],
      "manual_timings": {"0": [10.5, 12.5], "1": [20.5, 22.5]},
    }
    sess_mod.session_path(media).write_text(json.dumps(raw), encoding="utf-8")
    data = sess_mod.load(media)
    t = {e["text"]: (e["start"], e["end"]) for e in data["manual_edits"]["timings"]}
    assert t == {"alpha": (10.5, 12.5), "beta": (20.5, 22.5)}
    # migration in-memory only; file untouched until next save
    on_disk = json.loads(sess_mod.session_path(media).read_text(encoding="utf-8"))
    assert on_disk["manual_timings"] == raw["manual_timings"]
    assert "manual_edits" not in on_disk
