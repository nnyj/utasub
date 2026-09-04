"""Tests for prepare/finalize split and order-aware setlist DP.
prepare/finalize: headless behavior, merged cues, export gating.
DP: monotonicity (incl. low-sim in-order), skips, out-of-order rejection."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from utasub.core.regions import Region
from utasub.core.providers import Candidate
from utasub.core.place import AlignOpts

def _make_regions():
  return [Region(10, 250), Region(300, 550)]

def _make_segments():
  segs = []
  for r in _make_regions():
    for t in range(int(r.start), int(r.end), 10):
      segs.append((float(t), float(t + 5), f"word at {t}"))
  return segs

def _make_long_segments():
  """2 regions spanning >600s so the short-media gate does not fire."""
  segs = []
  for start, end in [(10, 250), (300, 650)]:
    for t in range(start, end, 10):
      segs.append((float(t), float(t + 5), f"word at {t}"))
  return segs

def _make_candidate(title, sim=0.5):
  c = Candidate(title=title, album="Album", artist="Artist", source="NetEase",
                duration_ms=240000, lrc="[00:01.00] line one\n[00:05.00] line two")
  return (sim, c)

def _make_cand(title, lrc="[00:01.00] line"):
  return Candidate(title=title, album="", artist="Art", source="NetEase",
                   duration_ms=240000, lrc=lrc)

def _fetch_by_query(titles):
  """Mocks fetch_from_providers: returns track whose title is in query.
  Patched at dispatch level, skips provider cache/network."""
  def mock_fetch(query, providers, fresh=False, limit=10, on_error=None):
    for title in titles:
      if title.lower() in query.lower():
        lrc = "" if title == "Overture" else "[00:01.00] line"
        return [_make_cand(title, lrc)]
    return []
  return mock_fetch

def _score_by_table(regions, segs, score_table):
  """Build a score_candidates mock that reverse-looks-up the region by segment slice."""
  from utasub.core.regions import region_segments

  def mock_score(candidates, segments, dur_ms):
    region_idx = next((i for i, r in enumerate(regions)
                       if segments == region_segments(segs, r)), None)
    if region_idx is None:
      return []
    result = [(score_table.get((region_idx, c.title), 0.0), c)
              for c in candidates if c.lrc]
    result.sort(key=lambda x: -x[0])
    return result
  return mock_score

def _region_segs(regions):
  return [(float(t), float(t + 5), f"w{t}")
          for r in regions for t in range(int(r.start), int(r.end), 10)]

def _run_dp(regions, tracklist, titles, score_table):
  """Run _setlist_dp_assign with table-driven provider + score mocks."""
  from utasub.core.multi_song import _setlist_dp_assign
  segs = _region_segs(regions)
  with patch("utasub.core.providers.fetch_from_providers", _fetch_by_query(titles)), \
       patch("utasub.core.align.score_candidates", _score_by_table(regions, segs, score_table)):
    assignments, _ = _setlist_dp_assign(
      regions, tracklist, segs, ["NetEase"], "Art", False, [])
  return assignments

# --- setlist DP ---

def test_dp_monotonic_including_low_sim_in_order():
  """DP picks the best monotonic matching; a low-sim (>=0.30) match is accepted in order."""
  regions = [Region(0, 240), Region(300, 540), Region(600, 840)]
  tracklist = [(1, "Song A"), (2, "Song B"), (3, "Song C")]
  # region 0 matches Song A only weakly (0.32) but in order → still accepted
  score_table = {
    (0, "Song A"): 0.32, (0, "Song B"): 0.30, (0, "Song C"): 0.20,
    (1, "Song A"): 0.20, (1, "Song B"): 0.70, (1, "Song C"): 0.30,
    (2, "Song A"): 0.10, (2, "Song B"): 0.20, (2, "Song C"): 0.90,
  }
  a = _run_dp(regions, tracklist, ["Song A", "Song B", "Song C"], score_table)
  assert a[0] is not None and a[0][1].title == "Song A"
  assert a[1] is not None and a[1][1].title == "Song B"
  assert a[2] is not None and a[2][1].title == "Song C"

def test_dp_skips_mc_region_and_lyricless_track():
  """DP may skip a region (MC) and a track (instrumental with no lyrics)."""
  regions = [Region(0, 240), Region(300, 540), Region(600, 840), Region(900, 1140)]
  tracklist = [(1, "Overture"), (2, "Song A"), (3, "Song B"), (4, "Song C"), (5, "Song D")]
  score_table = {
    (0, "Song A"): 0.75, (2, "Song B"): 0.80,
    (3, "Song C"): 0.60, (3, "Song D"): 0.85,
  }
  titles = ["Overture", "Song A", "Song B", "Song C", "Song D"]
  a = _run_dp(regions, tracklist, titles, score_table)
  assert a[0] is not None and a[0][1].title == "Song A"
  assert a[1] is None  # MC region unmatched
  assert a[2] is not None and a[2][1].title == "Song B"
  assert a[3] is not None and a[3][1].title == "Song D"

def test_dp_rejects_high_sim_out_of_order():
  """A high-sim out-of-order pair cannot both be assigned; DP keeps the single best."""
  regions = [Region(0, 240), Region(300, 540)]
  tracklist = [(1, "Song A"), (2, "Song B")]
  score_table = {
    (0, "Song A"): 0.20, (0, "Song B"): 0.90,
    (1, "Song A"): 0.85, (1, "Song B"): 0.10,
  }
  a = _run_dp(regions, tracklist, ["Song A", "Song B"], score_table)
  assert sum(1 for x in a if x is not None) == 1
  assert a[0] is not None and a[0][1].title == "Song B"
  assert a[1] is None

# --- finalize / prepare ---

def _finalize(regions, assignments, strategy="mock", export=True, tmpdir=None,
              on_chain=None):
  """Run _multi_song_finalize with run_chain mocked to echo the region's segments."""
  from utasub.core.multi_song import _multi_song_finalize
  segments = _make_segments()

  def mock_run_chain(lines, segs, audio, opts=None, **kw):
    if on_chain:
      on_chain()
    return list(segs), None, {"strategy": strategy}

  with patch("utasub.core.place.run_chain", mock_run_chain), \
       patch("lyrickit.classify_lines", return_value=["sung", "sung"]):
    path = Path(tmpdir) / "test.mkv"
    path.touch()
    return _multi_song_finalize(path, segments, None, regions, assignments,
                                romaji=False, opts=AlignOpts(use_fa=False),
                                export=export)

@pytest.mark.parametrize("sim,user_picked,guarded", [
  (0.40, False, True),   # sub-threshold coarse_envelope: fall back to ASR
  (0.40, True, False),   # the human vouched for the match: keep the placement
  (0.60, False, False),  # above threshold: nothing to guard
])
def test_finalize_low_confidence_guard(sim, user_picked, guarded):
  """Low-confidence guard replaces weak coarse_envelope placement with raw ASR,
  unless candidate was user-picked."""
  _, cand = _make_candidate("Song A", sim)
  cand.user_picked = user_picked
  with tempfile.TemporaryDirectory() as tmpdir:
    _, metas = _finalize([Region(10, 250)], [(sim, cand)],
                         strategy="coarse_envelope", tmpdir=tmpdir)
  fell_back = any(m.get("strategy") == "low_confidence_asr" for m in metas)
  assert fell_back is guarded, metas

def test_finalize_export_false_runs_chain_per_assigned_region_and_writes_nothing():
  """One run_chain call per assigned region; export=False writes no SRT/session."""
  regions = [Region(10, 250), Region(300, 550)]
  assignments = [_make_candidate("Song A", 0.6), None]
  calls = []
  with tempfile.TemporaryDirectory() as tmpdir:
    _, metas = _finalize(regions, assignments, export=False, tmpdir=tmpdir,
                         on_chain=lambda: calls.append(1))
    assert len(calls) == 1
    assert len(metas) == 2
    assert not list(Path(tmpdir).glob("*.srt"))
    assert not list(Path(tmpdir).glob("*.utasub.json"))

def test_single_region_align_reports_its_region_without_export():
  """Aligns one region, writes nothing."""
  from utasub.core.multi_song import _finalize_single_region
  region = Region(300, 550)
  segments = _make_segments()

  def mock_run_chain(lines, segs, audio, opts=None, **kw):
    return list(segs), None, {"strategy": "mock"}

  with tempfile.TemporaryDirectory() as tmpdir:
    with patch("utasub.core.place.run_chain", mock_run_chain), \
         patch("lyrickit.classify_lines", return_value=["sung", "sung"]):
      cues, meta = _finalize_single_region(
        segments, None, region, 1, _make_candidate("Song A", 0.6),
        opts=AlignOpts(use_fa=False))
    assert all(300 <= c[0] <= 550 for c in cues)
    assert meta.get("region_idx") == 1
    assert not list(Path(tmpdir).glob("*.srt"))

def test_pool_fallback_rejects_below_threshold():
  """Artist-pool fallback leaves regions unassigned when best sim < POOL_SIM_THRESHOLD."""
  from utasub.cli import _multi_song_prepare
  segments = _make_long_segments()
  cand = _make_candidate("Noise Song", 0.4)  # below 0.55

  def mock_fetch(query, providers, fresh=False, limit=10, on_error=None):
    return [cand[1]]

  with patch("utasub.core.providers.fetch_from_providers", mock_fetch), \
       patch("utasub.core.setlist.discover", return_value=([], "", "")):
    result = _multi_song_prepare(
      Path("test.mkv"), segments, None, ["NetEase"], False, use_setlist=False)
  assert result is not None
  _, _, assignments, _, _ = result
  assert all(a is None for a in assignments)

def test_headless_writes_srt_and_multi_song_session():
  """Runs prepare+finalize, writes srt + multi_song session."""
  from utasub.cli import _multi_song_headless
  segments = _make_segments()

  def mock_fetch(query, providers, fresh=False, limit=10, on_error=None):
    return [_make_candidate("Song A", 0.6)[1]]

  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "test.mkv"
    path.touch()
    with patch("utasub.core.providers.fetch_from_providers", mock_fetch), \
         patch("utasub.core.setlist.discover", return_value=([], "", "")), \
         patch("utasub.core.place.run_chain",
               side_effect=lambda l, s, a, opts=None, **kw: (list(s), None, {"strategy": "mock"})), \
         patch("lyrickit.classify_lines", return_value=["sung", "sung"]):
      _multi_song_headless(path, segments, None, ["NetEase"], False,
                           romaji=False, opts=AlignOpts(use_fa=False),
                           use_setlist=False)
    assert list(Path(tmpdir).glob("*.srt"))
    session_files = list(Path(tmpdir).glob("*.utasub.json"))
    assert session_files
    sess = json.loads(session_files[0].read_text(encoding="utf-8"))
    assert sess.get("placement", {}).get("multi_song")

def test_asr_straddling_a_span_is_dropped():
  """ASR segment mostly inside a placed span is dropped as duplicate lyrics;
  one fully outside a span is MC talk and survives."""
  from utasub.core.multi_song import _inside_frac, ASR_INSIDE_MAX
  spans = [(160.1, 400.0), (500.0, 700.0)]

  def kept(seg):
    return _inside_frac(seg, spans) < ASR_INSIDE_MAX

  # starts 0.59s before the span, too little to be "contained", 88% under lyrics
  assert not kept((159.5, 164.6, "head straddle"))
  # straddles the tail, most of it under the lyrics
  assert not kept((380.0, 402.0, "tail straddle"))
  # fully contained: covered by the lyrics
  assert not kept((200.0, 210.0, "contained"))
  # fully outside every span: MC talk between songs, must survive
  assert kept((410.0, 430.0, "between songs"))
  assert kept((720.0, 740.0, "after last song"))
  # only just clips a span edge: its bulk is outside, so it stays
  assert kept((395.0, 420.0, "mostly outside"))
  # a fragment sitting on the span edge is the song's own, not a separate line
  assert not kept((159.9, 160.4, "edge fragment"))

# --- suspect flagging, source provenance, asr fill ---

def _dp_with_scored(regions, tracklist, titles, score_table):
  """_setlist_dp_assign returning both assignments and per-region scored lists."""
  from utasub.core.multi_song import _setlist_dp_assign
  segs = _region_segs(regions)
  with patch("utasub.core.providers.fetch_from_providers", _fetch_by_query(titles)), \
       patch("utasub.core.align.score_candidates", _score_by_table(regions, segs, score_table)):
    return _setlist_dp_assign(regions, tracklist, segs, ["NetEase"], "Art",
                              False, [])

def test_dp_flags_a_weak_win_as_suspect():
  """A win under SUSPECT_SIM is a coin flip, region.suspect says so."""
  regions = [Region(0, 240), Region(300, 540)]
  tracklist = [(1, "Song A"), (2, "Song B")]
  score_table = {(0, "Song A"): 0.35, (0, "Song B"): 0.10,
                 (1, "Song A"): 0.10, (1, "Song B"): 0.80}
  _dp_with_scored(regions, tracklist, ["Song A", "Song B"], score_table)
  assert regions[0].suspect, "weak win not flagged"
  assert not regions[1].suspect, "confident win flagged"

def test_dp_flags_a_near_tie_and_names_the_runner_up():
  """Two titles within SUSPECT_MARGIN: the region needs a human look."""
  regions = [Region(0, 240)]
  tracklist = [(1, "Song A"), (2, "Song B")]
  score_table = {(0, "Song A"): 0.62, (0, "Song B"): 0.60}
  _dp_with_scored(regions, tracklist, ["Song A", "Song B"], score_table)
  assert regions[0].suspect
  assert regions[0].runner_up == "Song B", regions[0].runner_up

def test_dp_relabel_does_not_touch_the_shared_candidate():
  """The winning Candidate also sits in other regions' scored lists; the
  Setlist: relabel must land on a copy."""
  regions = [Region(0, 240), Region(300, 540)]
  tracklist = [(1, "Song A"), (2, "Song B")]
  score_table = {(0, "Song A"): 0.90, (0, "Song B"): 0.10,
                 (1, "Song A"): 0.20, (1, "Song B"): 0.85}
  assignments, scored = _dp_with_scored(regions, tracklist,
                                        ["Song A", "Song B"], score_table)
  assert assignments[0][1].source == "Setlist:Wikipedia"
  shared = [c for _, c in scored[1] if c.title == "Song A"]
  assert shared and shared[0].source == "NetEase", "provenance leaked to region 1"

def test_asr_fill_off_keeps_a_region_whose_lyrics_never_placed():
  """--no-asr-fill drops the transcript between songs. A region that placed no
  song span has only its transcript, so dropping that leaves it blank."""
  regions = [Region(10, 250)]
  with tempfile.TemporaryDirectory() as tmpdir:
    from utasub.core.multi_song import _multi_song_finalize
    segments = _make_segments()

    def mock_run_chain(lines, segs, audio, opts=None, **kw):
      return list(segs), None, {"strategy": "mock"}

    path = Path(tmpdir) / "test.mkv"
    path.touch()
    with patch("utasub.core.place.run_chain", mock_run_chain), \
         patch("lyrickit.classify_lines", return_value=["sung", "sung"]):
      kept, _ = _multi_song_finalize(path, segments, None, regions, [None],
                                     romaji=False, opts=AlignOpts(use_fa=False),
                                     export=False, mc=False, asr_fill=False)
  assert kept, "the unplaced region lost its transcript too"
  assert all(10 <= c[0] <= 250 for c in kept), "kept transcript outside every region"

def test_asr_fill_drops_junk_between_songs():
  """Crowd noise the ASR invented text over never reaches the subtitle track."""
  from utasub.core.multi_song import _multi_song_finalize
  segments = [(400.0, 402.0, "ありがとうございます、みなさん元気ですか"),
              (405.0, 406.0, "ワーワーワーワー"),
              (410.0, 422.0, "あー")]
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "test.mkv"
    path.touch()
    cues, _ = _multi_song_finalize(path, segments, None, [], [], romaji=False,
                                   opts=AlignOpts(use_fa=False), export=False,
                                   mc=False)
  texts = [c[2] for c in cues]
  assert "ワーワーワーワー" not in texts
  assert "あー" not in texts
  assert len(texts) == 1

def test_region_records_the_index_of_the_candidate_actually_picked():
  """candidate_idx points back into the region's scored list, so it has to name
  the winner, not the top score."""
  from utasub.core.multi_song import _align_one_region
  first, second = _make_cand("A", lrc=""), _make_cand("B", lrc="")
  scored = [(0.9, first), (0.8, second)]
  region = Region(0.0, 60.0)
  _cues, meta = _align_one_region(region, 0, [], None, (0.8, second),
                                  AlignOpts(), scored=scored)
  assert meta["strategy"] == "parse_empty"
  assert region.candidate_idx == 1

def test_region_candidate_index_defaults_to_zero_without_a_scored_list():
  from utasub.core.multi_song import _align_one_region
  region = Region(0.0, 60.0)
  _align_one_region(region, 0, [], None, (0.8, _make_cand("A", lrc="")), AlignOpts())
  assert region.candidate_idx == 0
