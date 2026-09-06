"""ASS karaoke fills, MC romaji over original, and the subtitle mux command."""
import re
import tempfile
from pathlib import Path

import pytest

from utasub.core.align import Cue
from utasub.core import export

def _tmp_media(tmpdir):
  path = Path(tmpdir) / "concert.mkv"
  path.touch()
  return path

def _spans(rom, per=0.2, at=0.0):
  """Token spans for a romaji string, one per non-space char, back to back."""
  out = []
  for c in rom:
    if not c.isspace():
      out.append((c.lower(), round(at, 3), round(at + per, 3)))
    at += per
  return out

# --- \k run building ---

def test_kf_runs_start_on_the_word_they_were_built_from():
  """Summing to the cue length is an identity of the run builder, true for any
  spans it accepts. What can break is which word each run pays for, so the
  cumulative time of every run is checked against its word's first span."""
  import re
  rom = "yume no tsuzuki"
  spans = _spans(rom, per=0.2)
  line = export._kf_line(rom, spans, cue_len=3.0)
  assert line is not None
  cs = [int(n) for n in re.findall(r"\\k(\d+)", line)]
  words = rom.split()
  assert len(cs) == len(words), (cs, line)  # one run per word, none leading
  idx, acc = 0, 0
  for word, n in zip(words, cs):
    assert acc == round(spans[idx][1] * 100), (acc, spans[idx][1], line)
    acc += n
    idx += len(word)
  assert acc == 300, (acc, line)

def test_kf_runs_keep_every_character_of_the_romaji():
  rom = "yume no tsuzuki"
  line = export._kf_line(rom, _spans(rom), cue_len=3.0)
  assert re.sub(r"\{[^}]*\}", "", line) == rom

def test_kf_leading_gap_becomes_an_empty_run():
  """Spans that start after the cue does mean the singer comes in late; the
  fill has to wait, not start under the first syllable."""
  rom = "abc"
  line = export._kf_line(rom, _spans(rom, per=0.1, at=0.5), cue_len=1.0)
  assert line.startswith("{\\k50}{\\k"), line

def test_kf_line_groups_by_whitespace_word():
  """_kf_line fills per whitespace word; Japanese mora come pre-split from the
  romanizer, so this path only ever sees word units (romaji, pinyin, English)."""
  import re
  rom = "ni hao ma"
  line = export._kf_line(rom, _spans(rom, per=0.1), cue_len=0.7)
  units = re.findall(r"\{[^}]*\}([^{]*)", line)
  assert units == ["ni ", "hao ", "ma"], units

def test_kf_gives_up_when_the_spans_do_not_match_the_romaji():
  """A retyped line or a different pinned locale must fall back to plain text,
  not paint a fill against the wrong syllables."""
  assert export._kf_line("yume", _spans("sora"), cue_len=1.0) is None
  assert export._kf_line("yu", _spans("yume"), cue_len=1.0) is None

# --- export_ass ---

def test_ass_emits_kf_only_for_cues_carrying_spans():
  cue = Cue(10.0, 13.0, "ゆめのつづき", "lrc")
  from utasub.core.romanize import romanize
  cue.token_spans = _spans(romanize(cue.text, locale="ja"), per=0.15)
  plain = Cue(20.0, 22.0, "ゆめのつづき", "lrc")
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), [cue, plain], offset=0.0, lead=0.0)
    lines = [l for l in out.read_text(encoding="utf-8").splitlines()
             if l.startswith("Dialogue:")]
  # the timed cue splits into two single-line events (romaji + kanji), both with
  # a \k fill; the plain cue stays one \N event with no fill on its romaji half
  single = [l for l in lines if "\\N" in l]
  assert len(single) == 1, lines
  assert "\\k" not in single[0].split("\\N")[0], single[0]
  fills = [l for l in lines if "\\N" not in l]
  assert len(fills) == 2 and all("\\k" in l for l in fills), fills

def test_ass_kf_runs_follow_the_spans_through_the_export():
  """Through export_ass a CJK line fills per syllable: every run must begin on a
  real token start, be monotone, and tile the whole cue."""
  import re
  from utasub.core.romanize import romanize
  cue = Cue(10.0, 14.0, "そらに浮かぶ月", "lrc")
  rom = romanize(cue.text, locale="ja")
  spans = _spans(rom, per=0.1)
  cue.token_spans = spans
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), [cue], offset=0.0, lead=0.0)
    body = [l for l in out.read_text(encoding="utf-8").splitlines()
            if l.startswith("Dialogue:")][0]
  romaji_line = body.split("\\N")[0]
  cs = [int(n) for n in re.findall(r"\\k(\d+)", romaji_line)]
  starts = {round(s * 100) for _c, s, _e in spans}
  acc = 0
  for n in cs:
    assert acc in starts, (acc, starts, romaji_line)  # run opens on a token
    acc += n
  assert acc == 400, (acc, romaji_line)  # tiles the 4.0s cue

def _kanji_event(lines, glyph):
  """The stacked original-line event, found by a glyph only it carries."""
  return [l for l in lines if glyph in l and "\\N" not in l][0]

def test_ass_original_line_gets_per_word_kanji_karaoke():
  """The original also fills, as its own stacked event: Japanese per word (a
  multi-glyph word one run), timed off the same romaji spans, not fill-dimmed."""
  import re
  from utasub.core.romanize import romanize
  cue = Cue(10.0, 14.0, "空に浮かぶ月", "lrc")
  cue.token_spans = _spans(romanize(cue.text, locale="ja"), per=0.1)
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), [cue], offset=0.0, lead=0.0)
    lines = [l for l in out.read_text(encoding="utf-8").splitlines()
             if l.startswith("Dialogue:")]
  orig = _kanji_event(lines, "空")
  assert len(lines) == 2 and "\\N" not in orig, lines  # own event, not stacked in \N
  # a karaoke'd original is only shrunk, never fill-dimmed, or its \k reveal
  # would run backwards (grey -> transparent instead of grey -> white)
  assert "\\1a&H" not in orig, orig
  units = re.findall(r"\{\\k\d+\}([^{]*)", orig)
  assert units == ["空", "に", "浮かぶ", "月"], units  # 浮かぶ stays one word

def test_ass_original_line_chinese_karaoke_is_per_glyph():
  import re
  cue = Cue(10.0, 13.0, "你好", "lrc")
  cue.token_spans = [("n", 0.0, 0.5), ("i", 0.5, 1.0),
                     ("h", 1.5, 2.0), ("a", 2.0, 2.5), ("o", 2.5, 3.0)]
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), [cue], offset=0.0, lead=0.0)
    lines = [l for l in out.read_text(encoding="utf-8").splitlines()
             if l.startswith("Dialogue:")]
  units = re.findall(r"\{\\k\d+\}([^{]*)", _kanji_event(lines, "你"))
  assert units == ["你", "好"], units  # one run per hanzi

def test_write_time_offset_carries_the_karaoke_spans():
  """offset/lead rebuild the Cue, and spans are relative to its start: dropped
  or unshifted spans desync the fill from the line it paints."""
  cue = Cue(10.0, 13.0, "hello", "lrc")
  cue.token_spans = [("h", 0.0, 0.5)]
  shifted = export.apply_offset([cue], -2.0)[0]
  assert shifted.start == 8.0
  assert shifted.token_spans == [("h", 0.0, 0.5)], "offset moved the spans"
  led = export.apply_lead([cue], 1.0)[0]
  assert led.start == 9.0
  assert led.token_spans == [("h", 1.0, 1.5)], "lead did not delay the fill"

def test_ass_mc_line_is_romaji_over_original():
  """Gap 10: patter gets romaji over the original in the grey MC style,
  while a lyric cue in the same export stays on Default."""
  cues = [Cue(10.0, 15.0, "ゆめのつづき", "coarse"),
          (100.0, 104.0, "みなさん、こんばんは。", "mc")]
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), cues, offset=0.0, lead=0.0)
    text = out.read_text(encoding="utf-8")
  assert "Style: MC," in text
  dialogue = [l for l in text.splitlines() if l.startswith("Dialogue:")]
  assert len(dialogue) == 2, dialogue
  assert ",Default,," in dialogue[0], dialogue[0]
  line = dialogue[1]
  assert ",MC,," in line
  assert "Minasan" in line, line
  assert "\\Nみなさん" in line, "MC line lost the original under the romaji"

# --- .ass header: style lines ---

ASS_HEADER_CASES = [
  # defaults: constants reach the Default line, SecondaryColour is the grey
  # unfilled half of a \k wipe (not the ASS default red), Translation declared
  (None, [f"Style: Default,{export.ASS_FONT},{export.ASS_FONT_SIZE},",
          f",100,100,0,0,1,{export.ASS_OUTLINE},{export.ASS_SHADOW},2,60,60,",
          export.ASS_UNSUNG_COLOUR,
          "Style: Translation,", export.ASS_TR_COLOUR]),
  # style knobs: BorderStyle 3, top alignment
  ({"font": "Meiryo", "size": 40, "outline": 5, "box": True, "pos": 8},
   ["Style: Default,Meiryo,40,", ",100,100,0,0,3,5,1,8,60,60,"]),
]

@pytest.mark.parametrize("style,substrings", ASS_HEADER_CASES)
def test_ass_header_style_lines(style, substrings):
  kw = {"style": style} if style else {}
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), [Cue(1.0, 2.0, "hi", None)],
                            offset=0.0, lead=0.0, **kw)
    head = out.read_text(encoding="utf-8").split("[Events]")[0]
  for sub in substrings:
    assert sub in head, sub

# --- embed_subs ---

def _mux_cmd(monkeypatch, source_subs=0):
  """Capture the ffmpeg command. source_subs: subtitle streams the probe of the
  source reports."""
  import json
  seen = {}

  class _Proc:
    returncode = 0
    stderr = b""
    stdout = json.dumps(
      {"streams": [{"index": i} for i in range(source_subs)]}).encode()

  def run(cmd, **kw):
    if cmd[0] == "ffmpeg":
      seen["cmd"] = cmd
    return _Proc()

  monkeypatch.setattr(export.subprocess, "run", run)
  return seen

def test_embed_subs_puts_the_ass_first_and_flags_it_default(monkeypatch):
  seen = _mux_cmd(monkeypatch)
  with tempfile.TemporaryDirectory() as tmp:
    media = _tmp_media(tmp)
    srt = media.with_suffix(".ja.srt")
    ass = media.with_suffix(".ass")
    srt.touch()
    ass.touch()
    out = export.embed_subs(media, [srt, ass])
  cmd = seen["cmd"]
  assert out.name == "concert.subbed.mkv"
  assert cmd.index(str(ass)) < cmd.index(str(srt)), "srt muxed ahead of the ass"
  assert ["-c:s:0", "ass"] == cmd[cmd.index("-c:s:0"):cmd.index("-c:s:0") + 2]
  assert ["-c:s:1", "srt"] == cmd[cmd.index("-c:s:1"):cmd.index("-c:s:1") + 2]
  assert cmd[cmd.index("-disposition:s:0") + 1] == "default"
  assert cmd[cmd.index("-disposition:s:1") + 1] == "0"

def test_embed_subs_counts_the_sources_own_subtitle_streams(monkeypatch):
  """`-map 0` copies the source's subtitles first, so on an already-subbed file
  the new tracks are output subtitles 2 and 3: indices counted from 0 tag and
  flag the pre-existing tracks instead."""
  seen = _mux_cmd(monkeypatch, source_subs=2)
  with tempfile.TemporaryDirectory() as tmp:
    media = _tmp_media(tmp)
    srt = media.with_suffix(".ja.srt")
    ass = media.with_suffix(".ass")
    srt.touch()
    ass.touch()
    export.embed_subs(media, [srt, ass])
  cmd = seen["cmd"]
  assert ["-c:s:2", "ass"] == cmd[cmd.index("-c:s:2"):cmd.index("-c:s:2") + 2]
  assert ["-c:s:3", "srt"] == cmd[cmd.index("-c:s:3"):cmd.index("-c:s:3") + 2]
  assert cmd[cmd.index("-disposition:s:2") + 1] == "default"
  assert cmd[cmd.index("-disposition:s:0") + 1] == "0", "source track left default"
  assert "-c:s:0" not in cmd, "transcoding a pre-existing subtitle track"
  assert "title=ASS" in cmd and "language=jpn" in cmd

def test_embed_subs_tags_a_bare_ass_with_the_paired_srt_language(monkeypatch):
  """<name>.ass carries no locale in its name, and it is the default track, so
  it would ship untagged."""
  seen = _mux_cmd(monkeypatch)
  with tempfile.TemporaryDirectory() as tmp:
    media = _tmp_media(tmp)
    srt = media.with_suffix(".ja.srt")
    ass = media.with_suffix(".ass")
    srt.touch()
    ass.touch()
    export.embed_subs(media, [srt, ass])
  cmd = seen["cmd"]
  assert cmd[cmd.index("-metadata:s:s:0") + 1] == "language=jpn", cmd

def test_embed_subs_skips_sidecars_that_are_not_on_disk(monkeypatch):
  import pytest
  seen = _mux_cmd(monkeypatch)
  with tempfile.TemporaryDirectory() as tmp:
    media = _tmp_media(tmp)
    ass = media.with_suffix(".ass")
    ass.touch()
    export.embed_subs(media, [ass, media.with_suffix(".srt")])
    assert seen["cmd"].count("-i") == 2
    with pytest.raises(RuntimeError):
      export.embed_subs(media, [media.with_suffix(".srt")])

def test_english_lyric_line_still_gets_a_fill():
  """A non-CJK line has no romaji, so the fill has to go on the line itself or
  half a bilingual song loses its karaoke."""
  cue = Cue(10.0, 13.0, "Trail blazed across my mind", "lrc")
  cue.token_spans = _spans(cue.text.lower(), per=0.1)
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), [cue], offset=0.0, lead=0.0)
    line = [l for l in out.read_text(encoding="utf-8").splitlines()
            if l.startswith("Dialogue:")][0]
  assert "\\k" in line, line
  assert "\\N" not in line, "English line got a dimmed duplicate"
  import re
  bare = re.sub(r"\{[^}]*\}", "", line.split(",", 9)[9])
  assert bare == cue.text, f"the fill altered the line: {bare}"

def test_spans_running_past_a_trimmed_cue_end_are_clamped():
  """The envelope can trim an end back inside the spans; an unclamped tail
  would wipe on after the line has left the screen."""
  import re
  rom = "abcd"
  line = export._kf_line(rom, _spans(rom, per=1.0), cue_len=2.0)
  total = sum(int(n) for n in re.findall(r"\\k(\d+)", line))
  assert total == 200, (total, line)

# --- gap 11: .lrc back into the curated library ---

def test_lrc_rebases_to_the_region_and_heads_with_the_candidate():
  from utasub.core.providers import Candidate
  cand = Candidate(title="Song A", album="", artist="Artist", source="NetEase",
                   duration_ms=0, lrc="")
  cues = [Cue(100.0, 103.0, "first line", "lrc"),
          Cue(104.5, 107.0, "second line", "lrc"),
          Cue(500.0, 502.0, "next song", "lrc")]
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_lrc(_tmp_media(tmp), cues, (100.0, 200.0),
                            candidate=cand)
    text = out.read_text(encoding="utf-8")
  assert out.name == "concert.lrc"
  assert text == ("[ti:Song A]\n[ar:Artist]\n"
                  "[00:00.00] first line\n[00:04.50] second line\n")

def test_lrc_goes_to_the_library_named_after_the_candidate():
  from utasub.core.providers import Candidate
  cand = Candidate(title="Song A", album="", artist="Artist", source="NetEase",
                   duration_ms=0, lrc="")
  with tempfile.TemporaryDirectory() as tmp:
    lib = Path(tmp) / "library"
    out = export.export_lrc(_tmp_media(tmp), [Cue(1.0, 2.0, "hi", "lrc")],
                            candidate=cand, out_dir=lib)
  assert out == lib / "Artist - Song A.lrc"

# --- per-cue export locale ---

def test_each_cue_is_romanized_in_its_own_language():
  """One locale over a merged concert rendered the Korean song as hangul and
  every Chinese line as literal `?`."""
  cues = [Cue(0.0, 2.0, "ゆめのつづき", "lrc"), Cue(2.0, 4.0, "ゆめのうた", "lrc"),
          Cue(4.0, 6.0, "사랑해 그대여", "lrc"), Cue(6.0, 8.0, "我愛你", "lrc")]
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), cues, offset=0.0, lead=0.0)
    body = out.read_text(encoding="utf-8").split("[Events]")[1].lower()
  assert "?" not in body
  assert "saranghae" in body
  assert "wo ai ni" in body
  assert "yume no tsuzuki" in body

def test_srt_name_follows_the_majority_language():
  with tempfile.TemporaryDirectory() as tmp:
    media = _tmp_media(tmp)
    ko_heavy = [(0.0, 2.0, "사랑해 그대여"), (2.0, 4.0, "밤편지 보내"),
                (4.0, 6.0, "ゆめのつづき")]
    assert export.srt_name(media, ko_heavy).name == "concert.ko.srt"
    assert export.srt_name(media, [(0.0, 2.0, "plain english")]).name == "concert.srt"

def test_export_srt_writes_a_romaji_sidecar_and_clamps_a_negative_start():
  """CJK cues get the second .romaji.srt track; an offset dragging the first cue
  before zero clamps to 0.0 and keeps its length, not a negative stamp."""
  with tempfile.TemporaryDirectory() as tmp:
    media = _tmp_media(tmp)
    written = export.export_srt(media, [(0.5, 2.0, "こんにちは")], offset=-1.0,
                                lead=0.0)
    assert [p.name for p in written] == ["concert.ja.srt", "concert.romaji.srt"]
    main, rom = (p.read_text(encoding="utf-8") for p in written)
  assert "00:00:00,000 --> 00:00:01,500" in main
  assert "kon" in rom.lower()

# --- translation line ---

def _ass_dialogue(cues, translations):
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), cues, offset=0.0, lead=0.0,
                            translations=translations)
    body = out.read_text(encoding="utf-8").split("[Events]")[1]
  return [ln for ln in body.splitlines() if ln.startswith("Dialogue:")]

def test_translation_adds_a_third_line_under_the_original():
  cues = [Cue(0.0, 2.0, "ゆめのつづき", "lrc"), Cue(2.0, 4.0, "あなたは夏の風", "lrc")]
  lines = _ass_dialogue(cues, {"ゆめのつづき": "梦的延续", "あなたは夏の風": "你似那夏日薰风"})
  assert [ln.count("\\N") for ln in lines] == [2, 2]
  assert "梦的延续" in lines[0] and "你似那夏日薰风" in lines[1]
  assert "{\\rTranslation}" in lines[0]

def test_no_translation_leaves_two_lines():
  cues = [Cue(0.0, 2.0, "ゆめのつづき", "lrc"), Cue(2.0, 4.0, "あなたは夏の風", "lrc")]
  lines = _ass_dialogue(cues, {})
  assert [ln.count("\\N") for ln in lines] == [1, 1]

def _margin_v(line):
  return int(line.split(",")[7])

def _ass_dialogue_top(cues, translations):
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), cues, offset=0.0, lead=0.0,
                            translations=translations, translation_top=True)
    body = out.read_text(encoding="utf-8").split("[Events]")[1]
  return [ln for ln in body.splitlines() if ln.startswith("Dialogue:")]

def test_translation_top_puts_the_translation_first_single_event():
  """translation_top=True renders the translation above the sung line in a single
  event: {\\rTranslation}<tr> leads, a bare \\r restores the cue style below."""
  cues = [Cue(0.0, 2.0, "ゆめのつづき", "lrc")]
  lines = _ass_dialogue_top(cues, {"ゆめのつづき": "梦的延续"})
  assert "{\\rTranslation}梦的延续\\N{\\r}" in lines[0], lines[0]

def test_translation_top_gets_the_highest_margin_v_when_stacked():
  """A karaoke'd original emits stacked events; translation_top lifts the
  Translation event to the highest MarginV so it sits above the sung line."""
  from utasub.core.romanize import romanize
  cue = Cue(10.0, 14.0, "空に浮かぶ月", "lrc")
  cue.token_spans = _spans(romanize(cue.text, locale="ja"), per=0.1)
  lines = _ass_dialogue_top([cue], {"空に浮かぶ月": "浮月"})
  tr_line = [l for l in lines if ",Translation,," in l][0]
  assert _margin_v(tr_line) == max(_margin_v(l) for l in lines), lines

def test_mc_cue_gets_a_translation_line():
  """MC patter used to skip the translation lookup; it now takes one too."""
  cues = [(100.0, 104.0, "みなさん、こんばんは。", "mc")]
  lines = _ass_dialogue(cues, {"みなさん、こんばんは。": "Hello everyone"})
  assert ",MC,," in lines[0]
  assert "{\\rTranslation}Hello everyone" in lines[0], lines[0]
