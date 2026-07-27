"""Credit-line cleanup wiring: parse_lrc output feeds lyrickit's detector.
Detector's own coverage (languages, watermarks, false positives) is
lyrickit's suite; this only checks shape agreement."""
from lyrickit import classify_lines, filter_credit_lines

from utasub.core.providers import parse_lrc


NETEASE_LRC = """\
[ti:天青色等烟雨]
[ar:周杰倫]
[al:我很忙]
[by:网易云音乐]
[00:00.00] 作词：方文山
[00:01.00] 作曲：周杰伦
[00:02.00]天青色等烟雨 而我在等你
[00:10.00]炊烟袅袅升起 隔江千万里
[00:35.00] February: the month of love
"""


def test_parsed_lrc_feeds_the_credit_detector():
  """classify_lines accepts parse_lrc's (time, text) rows; filter_credit_lines
  drops credits, keeps sung lines in order."""
  lines = parse_lrc(NETEASE_LRC)
  verdicts = classify_lines(lines)
  assert len(verdicts) == len(lines)

  struck = [t for (_, t), v in zip(lines, verdicts) if v == "credit"]
  assert any("作词" in t for t in struck), struck

  filtered = filter_credit_lines(lines, verdicts)
  assert len(filtered) < len(lines), "no lines filtered"
  texts = [t for _, t in filtered]
  assert any("天青色" in t for t in texts), texts
  times = [t for t, _ in filtered if t is not None]
  assert times == sorted(times), "order not preserved"
