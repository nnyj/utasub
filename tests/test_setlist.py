"""Unit tests for setlist discovery: extraction heuristics, pasted-setlist parsing,
discover flow (mocked HTTP), cache read/write, --setlist-file plumbing."""
import json
import tempfile

import pytest
from pathlib import Path
from unittest import mock

WIKI_NUMBERED = """\
== セットリスト ==
1. 星の欠片
2. 雨のあとで
3. 夜明けの唄
4. だから僕は歌を書いた
5. 待って。
6. カタルシス
7. ただ風に舞う
8. 夜歩
9. 雨とラムネ
10. 遠遠遠夜
"""

WIKI_TABLE = """\
{{Track listing
| title1 = 星の欠片
| length1 = 4:32
| title2 = 雨のあとで
| length2 = 3:55
| title3 = 夜明けの唄
| length3 = 4:10
| title4 = だから僕は歌を書いた
| length4 = 5:02
}}
"""

MESSY_WEB = """\
<div class="setlist-body">
Concert recap: Hoshikuzu Live 2024 at Aozora Hall
Setlist:
1. 星の欠片 (Hoshi no Kakera)
2. 雨のあとで (Ame no Ato de)
3. 夜明けの唄 (Yoake no Uta)
Some commentary about the performance here...
4. だから僕は歌を書いた
5. カタルシス
Encore:
6. ただ風に舞う
</div>
"""

NO_TRACKLIST = """\
This is a biography page about Hoshikuzu.
They are a Japanese band formed in 2017.
No setlist information here at all.
"""

def test_heuristic_extracts_wiki_numbered_list():
  from utasub.core.setlist import _extract_heuristic
  tracks = _extract_heuristic(WIKI_NUMBERED)
  assert len(tracks) == 10
  assert tracks[0][1] == "星の欠片"
  assert tracks[9][1] == "遠遠遠夜"

def test_heuristic_extracts_wiki_track_template():
  from utasub.core.setlist import _extract_heuristic
  tracks = _extract_heuristic(WIKI_TABLE)
  assert len(tracks) == 4
  assert tracks[0] == (1, "星の欠片")
  assert tracks[3] == (4, "だから僕は歌を書いた")

def test_heuristic_extracts_from_messy_web_page():
  from utasub.core.setlist import _extract_heuristic
  tracks = _extract_heuristic(MESSY_WEB)
  assert len(tracks) >= 5
  assert "星の欠片" in tracks[0][1]

def test_heuristic_returns_empty_when_no_tracklist():
  from utasub.core.setlist import _extract_heuristic
  assert _extract_heuristic(NO_TRACKLIST) == []

# --- parse_pasted ---

PASTE_CASES = [
  # plain one-per-line
  ("星の欠片\n雨のあとで\n夜明けの唄",
   ["星の欠片", "雨のあとで", "夜明けの唄"]),
  # numbered, three punctuation styles
  ("1. 星の欠片\n02 雨のあとで\n3) 夜明けの唄",
   ["星の欠片", "雨のあとで", "夜明けの唄"]),
  # bracketed numbering + trailing parenthetical kept
  ("[01] 星の欠片\n[02] 雨のあとで (Live ver.)",
   ["星の欠片", "雨のあとで (Live ver.)"]),
  # bullets
  ("- 星の欠片\n・雨のあとで\n• 夜明けの唄",
   ["星の欠片", "雨のあとで", "夜明けの唄"]),
  # comma-separated single line
  ("星の欠片, 雨のあとで, 夜明けの唄",
   ["星の欠片", "雨のあとで", "夜明けの唄"]),
  # slash / fullwidth-bar separated single line
  ("星の欠片 / 雨のあとで ／ 夜明けの唄｜カタルシス",
   ["星の欠片", "雨のあとで", "夜明けの唄", "カタルシス"]),
  # header + section junk + timestamps
  ("Setlist:\n1. 星の欠片 [00:12]\nEncore:\n2. 雨のあとで 01:23",
   ["星の欠片", "雨のあとで"]),
  # header on the same line as the first title
  ("セットリスト: 星の欠片\n雨のあとで", ["星の欠片", "雨のあとで"]),
  # dupes collapse, first position kept
  ("星の欠片\n雨のあとで\n星の欠片", ["星の欠片", "雨のあとで"]),
  ("", []),
]

@pytest.mark.parametrize("text,expected", PASTE_CASES)
def test_parse_pasted_formats(text, expected):
  from utasub.core.setlist import parse_pasted
  assert parse_pasted(text) == expected

def test_parse_pasted_keeps_comma_inside_a_lone_title():
  """One comma in a single line among many is part of the title, not a delimiter."""
  from utasub.core.setlist import parse_pasted
  titles = parse_pasted("1. Hello, Goodbye\n2. 雨のあとで\n3. 夜明けの唄")
  assert titles[0] == "Hello, Goodbye"

def test_setlist_file_titles_reach_the_dp():
  """--setlist-file: load_setlist_file -> _multi_song_prepare -> DP tracklist."""
  from utasub import cli
  from utasub.core.regions import Region
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "set.txt"
    path.write_text("1. Track A\n2. Track B\n", encoding="utf-8")
    assert cli.load_setlist_file(path) == ["Track A", "Track B"]

    seen = {}

    def fake_dp(regions, tracklist, *a, **kw):
      seen["tracklist"] = tracklist
      seen["source_label"] = kw.get("source_label")
      return [None] * len(regions), [[] for _ in regions]

    regions = [Region(0.0, 300.0), Region(300.0, 600.0)]
    segments = [(0.0, 5.0, "x"), (700.0, 1300.0, "y")]
    from utasub.core import multi_song
    with mock.patch.object(multi_song, "_setlist_dp_assign", side_effect=fake_dp), \
         mock.patch("utasub.core.regions.detect_regions", return_value=regions), \
         mock.patch("utasub.core.providers.media_tags", return_value={}), \
         mock.patch("utasub.core.setlist.discover") as disc:
      cli._multi_song_prepare(Path("concert.mkv"), segments, None, ["NetEase"],
                              False, mc=False,
                              setlist_titles=cli.load_setlist_file(path))
    disc.assert_not_called()
    assert seen["tracklist"] == [(1, "Track A"), (2, "Track B")]
    assert seen["source_label"] == "Setlist:Manual"

def test_discover_returns_tracklist_from_wikipedia():
  from utasub.core import setlist
  wiki_page = "== セットリスト ==\n" + "\n".join(f"{i+1}. Track{i+1}" for i in range(8))

  def fake_http_get(url, headers=None, timeout=10):
    if "action=query" in url:
      return json.dumps({"query": {"search": [{"title": "TestPage"}]}})
    if "action=parse" in url:
      return json.dumps({"parse": {"wikitext": {"*": wiki_page}}})
    return None

  with mock.patch.object(setlist, "_http_get", side_effect=fake_http_get), \
       mock.patch.object(setlist, "_cache_read", return_value=None), \
       mock.patch.object(setlist, "_cache_save"):
    tracks, source, method = setlist.discover(
      {"artist": "TestArtist", "keywords": ["live"]}, fresh=True)
  assert len(tracks) == 8
  assert "Wikipedia" in source
  assert method == "heuristic"

def test_discover_returns_empty_when_nothing_found():
  from utasub.core import setlist

  def fake_http_get(url, headers=None, timeout=10):
    if "action=query" in url:
      return json.dumps({"query": {"search": [{"title": "BioPage"}]}})
    if "action=parse" in url:
      return json.dumps({"parse": {"wikitext": {"*": NO_TRACKLIST}}})
    return None

  with mock.patch.object(setlist, "_http_get", side_effect=fake_http_get), \
       mock.patch.object(setlist, "_cache_read", return_value=None), \
       mock.patch.object(setlist, "_cache_save"):
    tracks, source, method = setlist.discover(
      {"artist": "Test", "keywords": ["live"]}, fresh=True)
  assert tracks == []
  assert method == "none"

def test_cache_write_then_read_roundtrip():
  from utasub.core import setlist
  hints = {"artist": "CacheTest", "keywords": ["live"]}
  with tempfile.TemporaryDirectory() as tmpdir:
    with mock.patch.object(setlist, "LYRICS_DIR", Path(tmpdir)):
      data = {"tracklist": [(1, "Track1"), (2, "Track2")],
              "source": "test", "method": "heuristic"}
      setlist._cache_save(hints, data)
      loaded = setlist._cache_read(hints)
      assert loaded is not None
      assert len(loaded["tracklist"]) == 2
      assert loaded["source"] == "test"

def test_hints_from_path_extracts_keywords_and_strips_live():
  from utasub.core.setlist import hints_from_path
  hints = hints_from_path("dl/Hoshikuzu Live - Kioku.mkv", artist_hint="Hoshikuzu")
  assert hints["artist"] == "Hoshikuzu"
  assert "Kioku" in hints["keywords"]
  assert "Live" not in hints["keywords"]
