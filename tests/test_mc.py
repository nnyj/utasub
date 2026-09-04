"""Tests for MC (spoken intermission) detection: per-segment classification,
span merging, region exclusion from song matching, mc cues in the export."""
import tempfile
from pathlib import Path
from unittest.mock import patch

from utasub.core.regions import Region
from utasub.core.place import AlignOpts
from utasub.core.mc import is_junk_cue

# invented Japanese, no real lyrics or artist names
SPOKEN = [
  (100.0, 104.0, "みなさん、こんばんは。今日は本当にありがとうございます。"),
  (105.0, 108.0, "えー、次の曲は、ちょっと静かな感じです。"),
  (109.0, 112.0, "はい、では、よろしくお願いします!"),
]
SUNG = [
  (10.0, 15.0, "とおいそらのむこうへ"),
  (16.0, 21.0, "きみのこえがきこえる"),
  (22.0, 27.0, "あのひのゆめをかさねて"),
]

def _sung_segments(start, end, step=10):
  """Sung-shaped segments (bare kana, slow rate) filling a span."""
  return [(float(t), float(t + 5), "ゆめのつづきをうたう")
          for t in range(int(start), int(end), step)]

def _spoken_segments(start, end, step=6):
  """Patter-shaped segments filling a span, long enough to be its own region."""
  return [(float(t), float(t + 4), SPOKEN[i % len(SPOKEN)][2])
          for i, t in enumerate(range(int(start), int(end), step))]

def test_classify_mc_separates_spoken_from_sung_shapes():
  """Punctuated polite patter flags MC, bare sung phrases do not."""
  from utasub.core.mc import classify_mc
  assert classify_mc(SPOKEN) == [True, True, True]
  assert classify_mc(SUNG) == [False, False, False]

def test_classify_mc_ignores_short_text():
  """Too-short text carries no signal and is never flagged."""
  from utasub.core.mc import classify_mc
  assert classify_mc([(1.0, 2.0, "はい。"), (3.0, 4.0, "")]) == [False, False]

def test_mc_spans_merge_across_small_gaps_only():
  """Adjacent MC segments merge; a long gap starts a new span."""
  from utasub.core.mc import mc_spans
  later = [(400.0, 404.0, "ありがとうございます、みなさん。")]
  spans = mc_spans(SPOKEN + SUNG + later)
  assert len(spans) == 2, spans
  assert spans[0] == (100.0, 112.0)
  assert spans[1][0] == 400.0

def test_strip_mc_removes_only_overlapping_segments():
  """strip_mc drops segments overlapping a span, keeps the rest."""
  from utasub.core.mc import mc_spans, strip_mc
  segments = SUNG + SPOKEN
  kept = strip_mc(segments, mc_spans(segments))
  assert kept == SUNG

def test_mc_cues_are_verbatim_and_tagged():
  """MC cues keep the ASR text and carry confidence 'mc'."""
  from utasub.core.mc import mc_cues
  cues = mc_cues(SPOKEN + SUNG)
  assert [c.text for c in cues] == [t for _, _, t in SPOKEN]
  assert all(c.confidence == "mc" for c in cues)

def test_region_query_excludes_mc_text():
  """A region query built with mc spans excluded loses the patter phrases."""
  from utasub.core.mc import mc_spans
  from utasub.core.regions import region_query
  segments = SUNG + SPOKEN
  spans = mc_spans(segments)
  assert "みなさん" in region_query(segments, Region(0, 200))
  clean = region_query(segments, Region(0, 200), exclude=spans)
  assert clean and not any(m in clean for m in ("みなさん", "ありがとう", "次の曲"))

def test_mark_mc_regions_flags_spoken_region_and_spares_scored_song():
  """Majority-spoken region marked mc; same shape spared when candidate scores
  above threshold."""
  from utasub.core.mc import mark_mc_regions
  segments = _sung_segments(10, 90) + SPOKEN
  regions = [Region(10, 90), Region(95, 130)]

  marked = [Region(r.start, r.end) for r in regions]
  assert mark_mc_regions(marked, segments) == 1
  assert marked[1].mc and not marked[0].mc

  spared = [Region(r.start, r.end) for r in regions]
  scored = [[], [(0.80, object())]]
  assert mark_mc_regions(spared, segments, region_scored=scored) == 0

def test_prepare_marks_mc_region_and_drops_its_assignment():
  """_multi_song_prepare flags the spoken region and never assigns it a song."""
  from utasub.cli import _multi_song_prepare
  from utasub.core.providers import Candidate
  segments = (_sung_segments(10, 250) + _spoken_segments(300, 380)
              + _sung_segments(450, 800))
  cand = Candidate(title="Song A", album="", artist="Artist", source="NetEase",
                   duration_ms=240000, lrc="[00:01.00] line one")

  def mock_fetch(query, providers, fresh=False, limit=10, on_error=None):
    return [cand]

  with patch("utasub.core.providers.fetch_from_providers", mock_fetch), \
       patch("utasub.core.setlist.discover", return_value=([], "", "")), \
       patch("utasub.core.align.score_candidates",
             side_effect=lambda cands, segs, dur=0: [(0.10, c) for c in cands]):
    result = _multi_song_prepare(Path("test.mkv"), segments, None, ["NetEase"],
                                False, use_setlist=False)
  assert result is not None
  regions, _, assignments, notes, _ = result
  mc_idx = [i for i, r in enumerate(regions) if r.mc]
  assert mc_idx, [r.to_dict() for r in regions]
  assert all(assignments[i] is None for i in mc_idx)
  assert any("mc" in n for n in notes)

def _finalize_with_mc(mc, tmpdir):
  """Run _multi_song_finalize over one sung region plus trailing patter."""
  from utasub.core.multi_song import _multi_song_finalize
  segments = _sung_segments(10, 250) + SPOKEN
  path = Path(tmpdir) / "test.mkv"
  path.touch()
  with patch("utasub.core.place.run_chain",
             side_effect=lambda l, s, a, opts=None, **kw: (list(s), None, {"strategy": "mock"})), \
       patch("lyrickit.classify_lines", return_value=["sung"]):
    return _multi_song_finalize(path, segments, None, [Region(10, 250)], [None],
                                romaji=False, opts=AlignOpts(use_fa=False),
                                export=False, mc=mc)

def test_finalize_emits_mc_cues_once_when_enabled():
  """mc=True tags patter cues; text not also emitted as raw ASR."""
  with tempfile.TemporaryDirectory() as tmpdir:
    cues, _ = _finalize_with_mc(True, tmpdir)
  tagged = [c for c in cues if len(c) > 3 and c[3] == "mc"]
  assert len(tagged) == len(SPOKEN)
  for _, _, text in SPOKEN:
    assert sum(1 for c in cues if c[2] == text) == 1

def test_finalize_without_mc_leaves_patter_untagged():
  """mc=False leaves patter as plain ASR, no mc tag."""
  with tempfile.TemporaryDirectory() as tmpdir:
    cues, _ = _finalize_with_mc(False, tmpdir)
  assert not [c for c in cues if len(c) > 3 and c[3] == "mc"]
  assert any(c[2] == SPOKEN[0][2] for c in cues)

def test_ass_export_styles_mc_cues_separately():
  """export_ass routes mc cues to the MC style, lyric cues to Default."""
  from utasub.core.align import Cue
  from utasub.core.export import export_ass
  cues = [Cue(10.0, 15.0, "ゆめのつづき", "coarse"),
          Cue(100.0, 104.0, SPOKEN[0][2], "mc")]
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "test.mkv"
    path.touch()
    out = export_ass(path, cues, offset=0.0, lead=0.0)
    text = out.read_text(encoding="utf-8")
  assert "Style: MC," in text
  dialogue = [l for l in text.splitlines() if l.startswith("Dialogue:")]
  assert len(dialogue) == 2
  assert ",Default,," in dialogue[0]
  assert ",MC,," in dialogue[1]
  assert "Minasan" in dialogue[1], "MC line kept its kana instead of romaji"
  assert SPOKEN[0][2] not in dialogue[1], "MC line kept a dimmed original"

def test_session_roundtrips_mc_spans():
  """mc spans persist through save/load."""
  from utasub.core import session
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "test.mkv"
    path.touch()
    session.save(path, cues=[(10.0, 15.0, "line")], mc_spans=[(100.0, 112.0)])
    assert session.load(path)["mc_spans"] == [(100.0, 112.0)]

def test_is_junk_cue_flags_text_with_nothing_to_read():
  """Empty or punctuation-only text is never a line."""
  assert is_junk_cue((1.0, 3.0, "!!"))
  assert is_junk_cue((1.0, 3.0, "  "))

def test_is_junk_cue_keeps_short_real_lyric_lines():
  """Shortness alone dropped 嗚呼 / Oh / ラララ, all of them sung. It takes a
  second signal (a degenerate rate or duration) to call one junk."""
  assert not is_junk_cue((10.0, 12.0, "嗚呼"))
  assert not is_junk_cue((10.0, 11.5, "Oh"))
  assert not is_junk_cue((10.0, 11.5, "ラララ"))
  assert not is_junk_cue((10.0, 12.0, "ずっと"))
  assert is_junk_cue((10.0, 10.2, "ずっと")), "short + a degenerate stamp is junk"

def test_is_junk_cue_keeps_a_line_sustained_over_a_long_stamp():
  """A held line scores under the low chars/sec bar, which alone is not junk."""
  assert not is_junk_cue((100.0, 120.0, "きみのなまえを"))

def test_is_junk_cue_flags_crowd_noise_long_span():
  """A 12-second segment carrying 3 chars is crowd noise, not a line."""
  assert is_junk_cue((200.0, 212.0, "うおー"))

def test_is_junk_cue_flags_repeated_char_and_token_cheers():
  """Single repeated char and repeated short tokens are cheers."""
  assert is_junk_cue((10.0, 14.0, "ワーワーワーワー"))
  assert is_junk_cue((10.0, 14.0, "ah ah ah ah"))
  assert is_junk_cue((10.0, 14.0, "ーーーーー"))

def test_is_junk_cue_flags_garbage_fast_rate():
  """Chars/sec far above sung/spoken range is transcription garbage."""
  assert is_junk_cue((10.0, 10.8, "あいうえおかきくけこさしすせそたちつてと"))

def test_is_junk_cue_flags_repeat_of_prev_kept_text():
  """Normalized equality, punctuation included, counts as a repeat."""
  assert is_junk_cue((10.0, 13.0, "アンコール"), prev_text="アンコール")
  assert is_junk_cue((10.0, 13.0, "アンコール!"), prev_text="アンコール")

def test_is_junk_cue_keeps_a_real_line_that_starts_like_the_one_before():
  """A prefix overlap only reads as a repeat between two chants: ありがとう
  after ありがとうございました is the next lyric, not an echo."""
  assert not is_junk_cue((10.0, 13.0, "ありがとう"),
                         prev_text="ありがとうございました")

def test_is_junk_cue_keeps_real_short_sung_line_and_mc_patter():
  """Positive controls: real short sung line and MC patter survive."""
  assert not is_junk_cue(SUNG[0])
  assert not is_junk_cue((100.0, 104.0, "とおいそらへ"))
  assert not is_junk_cue(SPOKEN[0])

def test_filter_junk_drops_chants_cheers_keeps_songs_and_patter():
  """filter_junk threads prev kept text: encore chant repeated 4x collapses,
  cheers and crowd noise go, real sung lines and MC patter stay."""
  from utasub.core.mc import filter_junk
  segments = [
    SUNG[0],
    (30.0, 32.0, "アンコール"),
    (33.0, 35.0, "アンコール"),
    (36.0, 38.0, "アンコール"),
    (39.0, 41.0, "アンコール"),
    (50.0, 54.0, "ワーワーワーワー"),
    (60.0, 72.0, "うおー"),
    SPOKEN[0],
    SUNG[1],
  ]
  kept = filter_junk(segments)
  assert SUNG[0] in kept and SUNG[1] in kept and SPOKEN[0] in kept
  assert [k for k in kept if k[2] == "アンコール"] == [(30.0, 32.0, "アンコール")]
  assert not any(k[2] in ("ワーワーワーワー", "うおー") for k in kept)
