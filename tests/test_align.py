"""Unit tests for coarse alignment + candidate scoring.
Fixture: synthetic timed lines, replayed as ASR segments with jitter and misheard noise."""
import random
import pytest

_LYRICS = [
  "morning light on the empty street",
  "i left my shoes beside the door",
  "the kettle sings a tired song",
  "and nobody is listening",
  "we counted trains until the dawn",
  "you said the winter would be short",
  "it wasn't short at all",
  "the radio kept playing static",
  "i wrote your name on frozen glass",
  "it melted into nothing much",
  "so here we are again",
  "walking backwards through the years",
  "the map was drawn in pencil",
  "somebody erased the middle part",
  "i remember every wrong turn",
  "and none of the right ones",
  "hold the chorus like a stone",
  "throw it at the quiet water",
  "watch the circles disagree",
  "then settle into glass again",
  "there is a room i never open",
  "the dust has made a home in it",
  "i keep the key inside a book",
  "a book i promised not to read",
  "summer came in through the window",
  "left the curtains breathing slow",
  "i almost said your name out loud",
  "the ceiling fan said it for me",
  "counting streetlights on the way",
  "the last one always flickers",
  "as if it knows my name",
  "put the record on once more",
  "let the needle find the scratch",
  "we can sing it out of tune",
  "nobody is keeping score",
]

# uneven but plausible line gaps, cycled across the lyric lines
_GAPS = [4.2, 3.6, 5.1, 4.8, 3.9, 6.4, 4.0, 4.5]

def _timed_lines(start=8.4):
  """Synthetic timed LRC lines: [(seconds, text)]."""
  lines = []
  t = start
  for i, text in enumerate(_LYRICS):
    lines.append((round(t, 2), text))
    t += _GAPS[i % len(_GAPS)]
  return lines

def _lrc_text(timed_lines):
  """Render timed lines as LRC source."""
  return "".join(f"[{int(t // 60):02d}:{t % 60:05.2f}]{text}\n"
                 for t, text in timed_lines)

def garble_text(text, ratio=0.15):
  """Introduce misheard-text noise: swap ~ratio of chars to random ASCII."""
  chars = list(text)
  n = max(1, int(len(chars) * ratio))
  for _ in range(n):
    i = random.randint(0, len(chars) - 1)
    chars[i] = chr(random.randint(ord('a'), ord('z')))
  return "".join(chars)

def synthesize_asr_segments(timed_lines, jitter_s=0.3):
  """Build fake ASR segments from timed lines with jitter and garbled text."""
  random.seed(42)
  segments = []
  for i, (t, text) in enumerate(timed_lines):
    j = random.uniform(-jitter_s, jitter_s)
    start = max(0.0, t + j)
    if i + 1 < len(timed_lines):
      end = timed_lines[i + 1][0] + random.uniform(-jitter_s, jitter_s)
    else:
      end = start + 3.0
    end = max(start + 0.1, end)
    segments.append((start, end, garble_text(text)))
  return segments

def test_coarse_alignment_lands_matched_lines_within_1s():
  """Starts land within 1s of LRC time for matched lines; cue starts non-decreasing."""
  from utasub.core.align import coarse_map, place_by_envelope
  timed = _timed_lines()
  segments = synthesize_asr_segments(timed)
  texts = [t for _, t in timed]
  result = coarse_map(segments, texts)
  assert result is not None, "coarse_map returned None"
  starts, matched = result
  assert len(starts) == len(timed)

  within_1s = 0
  total_matched = 0
  for j, (lrc_t, _) in enumerate(timed):
    if not matched[j]:
      continue
    total_matched += 1
    if abs(starts[j] - lrc_t) <= 1.0:
      within_1s += 1
  assert within_1s >= total_matched * 0.7, \
    f"only {within_1s}/{total_matched} within 1s"

  cues = place_by_envelope(starts, texts, None, [])
  cue_starts = [s for s, _, _ in cues]
  for i in range(1, len(cue_starts)):
    assert cue_starts[i] >= cue_starts[i - 1], \
      f"non-decreasing violated at line {i}"

def test_candidate_scoring_prefers_correct_lyrics():
  """Good candidate scores higher than garbage and clears 0.4."""
  from utasub.core.align import score_candidate
  from utasub.core.providers import Candidate
  timed = _timed_lines()
  segments = synthesize_asr_segments(timed)
  dur_ms = int((timed[-1][0] + 4.0) * 1000)

  good = Candidate(title="Synth Song", album="", artist="Test Artist",
                   source="test", duration_ms=dur_ms, lrc=_lrc_text(timed))
  bad = Candidate(title="Wrong", album="", artist="Nobody",
                  source="test", duration_ms=dur_ms,
                  lrc="[00:01.00] This is a completely different song\n"
                      "[00:05.00] With unrelated lyrics\n")

  good_score = score_candidate(good, segments, dur_ms)
  bad_score = score_candidate(bad, segments, dur_ms)
  assert good_score > bad_score, f"good ({good_score}) should beat bad ({bad_score})"
  assert good_score > 0.4, f"good score too low: {good_score}"

# --- anchor-based warp ---

@pytest.mark.parametrize("durations", [None, [1.0, 1.0]])
def test_close_starts_keep_cues_nonoverlapping(durations):
  from utasub.core.align import build_cues
  cues = build_cues([1.0, 1.05], ["first", "second"], [], durations=durations)
  assert cues[0].start < cues[0].end <= cues[1].start
  assert cues[1].start < cues[1].end

def test_envelope_trim_floored_when_voice_lost_mid_phrase():
  """Trim far below word need means voice lost mid-phrase; LRC interval takes over."""
  from utasub.core.align import build_cues
  texts = ["ながいことばのれんしゅう"] * 6
  starts = [i * 10.0 for i in range(6)]
  durations = [6.0] * 6
  # every line voiced full 6s except line 3, stem drops after 1s
  runs = [(s, s + 6.0) for s in starts]
  runs[3] = (starts[3], starts[3] + 1.0)
  cues = build_cues(starts, texts, runs, durations=durations)
  assert cues[3].end - cues[3].start > 4.0, (
    f"envelope kept a {cues[3].end - cues[3].start:.2f}s cue for a 6s phrase")

  # trim to a plausible share of the phrase is left alone
  runs[3] = (starts[3], starts[3] + 4.0)
  cues = build_cues(starts, texts, runs, durations=durations)
  assert abs((cues[3].end - cues[3].start) - 4.0) < 0.6, "sound trim overridden"

# --- shared song locale, empty-key guard ---

def test_kanji_only_transcript_scores_as_japanese_not_pinyin():
  """The transcript alone detects as zh, so its pinyin was compared against the
  lyrics' romaji and a correct match landed under every threshold."""
  from utasub.core.align import score_candidate
  from utasub.core.providers import Candidate
  cand = Candidate(title="T", album="", artist="A", source="test", duration_ms=0,
                   lrc="[00:01.00] 歌詞は完全な漢字\n"
                       "[00:05.00] 明日の空を見上げて\n"
                       "[00:09.00] 今日の風が聞こえる")
  segments = [(0.0, 4.0, "今日 明日 空 見上 完全 漢字"), (4.0, 8.0, "風 聞 歌詞")]
  assert score_candidate(cand, segments, 0) > 0.3

def test_a_romanizer_that_returns_nothing_scores_zero_not_one():
  """Empty keys compare equal, so a missing romanizer tied every candidate at
  1.0 and the pick fell to duration alone."""
  from unittest.mock import patch
  from utasub.core import align
  from utasub.core.providers import Candidate
  cand = Candidate(title="T", album="", artist="A", source="test",
                   duration_ms=0, lrc="[00:01.00] 사랑해 그대여")
  segments = [(0.0, 3.0, "사랑해 그대여")]
  with patch.object(align, "romanize", lambda text, locale=None: ""):
    assert align.score_candidate(cand, segments, 0) == 0.0
    assert align.score_candidates([cand], segments, 0) == [(0.0, cand)]
