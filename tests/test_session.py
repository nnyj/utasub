"""Session save/load round-trip, deterministic re-export."""
import pytest

from utasub.core.providers import Candidate
from utasub.core import session
from utasub.core import export as export_mod

def _media(tmp_path):
  media = tmp_path / "test_song.mkv"
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

def test_round_trip_preserves_all_fields(tmp_path):
  media = _media(tmp_path)
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

def test_load_missing_returns_none(tmp_path):
  assert session.load(_media(tmp_path)) is None

def test_deterministic_reexport_japanese(tmp_path):
  """.ja.srt, .romaji.srt and .ass byte-identical across runs."""
  media = _media(tmp_path)
  cues = [(0.0, 3.0, "そらに浮かぶ月"), (3.0, 6.0, "風の声が聞こえる")]
  session.save(media, cues=cues, romaji=True)
  sess = session.load(media)

  bytes1 = [p.read_bytes() for p in session.reexport(media, sess)]
  bytes2 = [p.read_bytes() for p in session.reexport(media, sess)]
  assert len(bytes1) == len(bytes2) == 3
  for b1, b2 in zip(bytes1, bytes2):
    assert b1 == b2

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

# --- gap 7: hand-inserted lines survive a re-align ---

def test_added_line_is_replayed_when_the_aligner_cannot_rebuild_it():
  """A live-only chorus has no LRC source, so it is replayed verbatim."""
  from utasub.core.session import apply_manual_edits
  edits = {"timings": [], "deleted": [],
           "added": [{"text": "extra chorus", "at": 50.0, "start": 50.0,
                      "end": 52.0, "text_raw": "Extra Chorus"}]}
  fresh = [_c(10.0, 12.0, "alpha"), _c(90.0, 92.0, "omega")]
  out = apply_manual_edits(fresh, edits)
  assert [c.text for c in out] == ["alpha", "Extra Chorus", "omega"], "insert lost"
  assert (out[1].start, out[1].end) == (50.0, 52.0)
  assert out[1].confidence == "manual"

def test_added_line_is_not_duplicated_once_the_aligner_produces_it():
  """The next pass placing that text means it was never an insert."""
  from utasub.core.session import apply_manual_edits
  edits = {"timings": [], "deleted": [],
           "added": [{"text": "extra chorus", "at": 50.0, "start": 50.0,
                      "end": 52.0, "text_raw": "extra chorus"}]}
  fresh = [_c(10.0, 12.0, "alpha"), _c(50.4, 52.4, "extra chorus")]
  out = apply_manual_edits(fresh, edits)
  assert len(out) == 2, [c.text for c in out]

def test_added_survives_the_session_round_trip(tmp_path):
  media = _media(tmp_path)
  edits = {"timings": [], "deleted": [],
           "added": [{"text": "solo", "at": 5.0, "start": 5.0, "end": 6.0,
                      "text_raw": "Solo"}]}
  session.save(media, cues=_sample_cues(), manual_edits=edits)
  assert session.load(media)["manual_edits"]["added"] == edits["added"]

def test_manual_edits_always_load_with_all_three_lists(tmp_path):
  """Callers index the three keys directly, an old session has none of them."""
  media = _media(tmp_path)
  session.save(media, cues=_sample_cues(), manual_edits={"timings": []})
  got = session.load(media)["manual_edits"]
  assert set(got) >= {"timings", "deleted", "added"}

# --- gap 5 / gap 3: per-cue CTC evidence ---

def test_cue_extras_do_not_widen_the_serialized_cue(tmp_path):
  """Extras ride a parallel list, so the cue rows stay 3/4 elements wide."""
  import json
  from utasub.core.align import Cue
  media = _media(tmp_path)
  cue = Cue(1.0, 3.0, "hello", "lrc")
  cue.score = 0.5
  session.save(media, cues=[cue])
  raw = json.loads(session.session_path(media).read_text(encoding="utf-8"))
  assert raw["cues"] == [[1.0, 3.0, "hello", "lrc"]]
  assert raw["cue_meta"] == [{"score": 0.5}]

# --- gap 12: media stored as a basename ---

def test_media_is_stored_as_a_basename(tmp_path):
  import json
  media = _media(tmp_path)
  session.save(media, cues=_sample_cues())
  raw = json.loads(session.session_path(media).read_text(encoding="utf-8"))
  assert raw["media"] == "test_song.mkv", raw["media"]

def test_renamed_media_still_loads_and_says_so(tmp_path, capsys):
  """The ASR block and the manual edits are the expensive half; a rename must
  warn, not orphan them."""
  media = _media(tmp_path)
  session.save(media, cues=_sample_cues())
  moved = media.parent / "renamed.mkv"
  session.session_path(media).rename(session.session_path(moved))
  moved.write_bytes(b"\x00")
  loaded = session.load(moved)
  assert loaded is not None and len(loaded["cues"]) == 3
  assert "test_song" in capsys.readouterr().out

# --- gap 12: the write must not truncate on a crash ---

def test_write_is_atomic_and_leaves_no_temp_file(tmp_path, monkeypatch):
  def boom(*_a):
    raise OSError("disk full")

  media = _media(tmp_path)
  session.save(media, cues=_sample_cues())
  before = session.session_path(media).read_bytes()
  monkeypatch.setattr(session.os, "replace", boom)
  with pytest.raises(OSError):
    session.save(media, cues=[(0.0, 1.0, "clobbered")])
  assert session.session_path(media).read_bytes() == before, "session truncated"
  # the glob only means anything after a write that failed: before one, there
  # is nothing to leave behind
  assert not list(media.parent.glob("*.tmp")), "temp file left behind"

def test_two_media_files_sharing_a_stem_are_reported(tmp_path, capsys):
  """song.mkv and song.flac in one folder map to one session file; the stored
  name is the only thing that says which of them wrote it."""
  media = _media(tmp_path)
  session.save(media, cues=_sample_cues())
  sibling = media.with_suffix(".flac")
  sibling.write_bytes(b"\x00")
  session.load(sibling)
  assert "test_song.mkv" in capsys.readouterr().out, "stem collision went unsaid"

# --- per-cue extras ---

def test_cue_extras_survive_every_rebuild(tmp_path):
  """score and token_spans are the whole karaoke + Conf feature. Every rebuild
  site goes through dataclasses.replace, so they have to be Cue fields."""
  from dataclasses import replace
  from utasub.core.align import Cue
  cue = Cue(10.0, 13.0, "line", "lrc", score=-0.4,
            token_spans=[("a", 0.0, 0.5)])
  moved = replace(cue, start=20.0, end=23.0)
  assert (moved.score, moved.token_spans) == (-0.4, [("a", 0.0, 0.5)])

  media = _media(tmp_path)
  session.save(media, cues=[cue, Cue(20.0, 21.0, "plain", "lrc")])
  loaded = session.load(media)["cues"]
  assert loaded[0].score == -0.4 and loaded[0].token_spans == [["a", 0.0, 0.5]]
  assert getattr(loaded[1], "score", None) is None, "extras leaked onto a bare cue"
  edits = {"timings": [{"text": "line", "at": 10.0, "start": 11.0, "end": 14.0}]}
  retimed = session.apply_manual_edits([cue], edits)[0]
  assert retimed.start == 11.0
  assert (retimed.score, retimed.token_spans) == (-0.4, [("a", 0.0, 0.5)])

def test_an_inserted_repeat_of_an_existing_lyric_is_replayed():
  """An added record whose text the placement produces elsewhere in the song is
  still an insertion; only one made at about the same time is a text edit."""
  from utasub.core.align import Cue
  cues = [Cue(10.0, 12.0, "chorus", "lrc"), Cue(60.0, 62.0, "verse", "lrc")]
  edits = {"added": [{"text": "chorus", "at": 90.0, "start": 90.0, "end": 92.0,
                      "text_raw": "chorus"}]}
  out = session.apply_manual_edits(cues, edits)
  assert [c.text for c in out].count("chorus") == 2, "the extra chorus was dropped"
  same = {"added": [{"text": "chorus", "at": 10.0, "start": 10.0, "end": 12.0,
                     "text_raw": "chorus"}]}
  assert len(session.apply_manual_edits(cues, same)) == 2, "text edit replayed as insert"

# --- candidates ---

def test_user_picked_survives_the_round_trip(tmp_path):
  """The low-confidence guard reads user_picked, and lyrickit's Candidate does
  not declare it, so asdict alone drops it."""
  media = _media(tmp_path)
  cands = _sample_candidates()
  cands[1].user_picked = True
  session.save(media, candidates=cands, chosen_index=1, cues=_sample_cues())
  loaded = session.load(media)["candidates"]
  assert loaded[1].user_picked is True
  assert loaded[0].user_picked is False

def test_session_ass_style_reproduces_the_header(tmp_path):
  """The style knobs live in the session, so a re-export writes the same
  header without the flags that chose them."""
  media = _media(tmp_path)
  style = {"font": "Meiryo", "size": 40, "outline": 5, "box": True, "pos": 8}
  session.save(media, cues=[(0.0, 3.0, "hello")], romaji=False,
               ass_style=style)
  sess = session.load(media)
  assert sess["ass_style"] == style
  ass = next(p for p in session.reexport(media, sess) if p.suffix == ".ass")
  head = ass.read_text(encoding="utf-8").split("[Events]")[0]
  assert "Style: Default,Meiryo,40," in head
  assert f",100,100,0,0,3,5,{export_mod.ASS_SHADOW},8,100,100," in head

def test_translations_round_trip_and_reach_the_ass(tmp_path):
  media = _media(tmp_path)
  session.save(media, cues=[(0.0, 2.0, "ゆめ", "lrc")],
               translations={"ゆめ": "梦"})
  data = session.load(media)
  assert data["translations"] == {"ゆめ": "梦"}
  ass = next(p for p in session.reexport(media, data) if p.suffix == ".ass")
  assert "{\\rTranslation}" in ass.read_text(encoding="utf-8")

def test_plain_save_preserves_stored_translations(tmp_path):
  media = _media(tmp_path)
  session.save(media, cues=[(0.0, 2.0, "ゆめ", "lrc")], translations={"ゆめ": "梦"})
  session.save(media, cues=[(0.0, 2.0, "ゆめ", "lrc")])  # no translations arg
  assert session.load(media)["translations"] == {"ゆめ": "梦"}
