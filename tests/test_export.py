"""ASS karaoke fills, MC romaji, and the subtitle mux command."""
import re
import tempfile
from pathlib import Path

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

# --- \kf run building ---

def test_kf_runs_start_on_the_span_they_were_built_from():
  """Summing to the cue length is an identity of the run builder, true for any
  spans it accepts. What can break is which syllable each run pays for, so the
  cumulative time of every run is checked against its own span."""
  import re
  rom = "yume no tsuzuki"
  spans = _spans(rom, per=0.2)
  line = export._kf_line(rom, spans, cue_len=3.0)
  assert line is not None
  cs = [int(n) for n in re.findall(r"\\kf(\d+)", line)]
  assert len(cs) == len(spans), (cs, line)  # no leading wait: first span at 0.0
  acc = 0
  for n, (_c, start, _end) in zip(cs, spans):
    assert acc == round(start * 100), (acc, start, line)
    acc += n
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
  assert line.startswith("{\\kf50}{\\kf"), line

def test_kf_untimed_characters_ride_the_run_before_them():
  """Whitespace is dropped by the aligner, so it cannot own a run of its own."""
  rom = "ab cd"
  line = export._kf_line(rom, _spans(rom, per=0.1), cue_len=0.5)
  assert line.count("{\\kf") == 4, line  # 4 timed chars, no run for the space
  assert "}b {" in line, "the space did not stay attached to the b run"

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
  assert "\\kf" in lines[0], lines[0]
  assert "\\kf" not in lines[1], lines[1]

def test_ass_kf_runs_follow_the_spans_through_the_export():
  """Through export_ass the fill must still sit on the syllables the aligner
  timed, not merely add up to the cue."""
  import re
  from utasub.core.romanize import romanize
  cue = Cue(10.0, 14.0, "そらに浮かぶ月", "lrc")
  spans = _spans(romanize(cue.text, locale="ja"), per=0.1)
  cue.token_spans = spans
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), [cue], offset=0.0, lead=0.0)
    body = [l for l in out.read_text(encoding="utf-8").splitlines()
            if l.startswith("Dialogue:")][0]
  romaji_line = body.split("\\N")[0]
  cs = [int(n) for n in re.findall(r"\\kf(\d+)", romaji_line)]
  acc = 0
  for n, (_c, start, _end) in zip(cs, spans):
    assert acc == round(start * 100), (acc, start, romaji_line)
    acc += n
  assert acc == 400, (acc, romaji_line)

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

def test_ass_mc_line_is_romaji_only():
  """Gap 10: patter gets romaji in the grey MC style, with no dimmed original."""
  cue = (100.0, 104.0, "みなさん、こんばんは。", "mc")
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), [cue], offset=0.0, lead=0.0)
    text = out.read_text(encoding="utf-8")
  line = [l for l in text.splitlines() if l.startswith("Dialogue:")][0]
  assert ",MC,," in line
  assert "Minasan" in line, line
  assert "みなさん" not in line, line
  assert "\\N" not in line, "MC line got a second line"

def test_ass_unsung_colour_is_grey_not_the_default_red():
  """SecondaryColour is the unfilled half of a \\kf wipe."""
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), [Cue(1.0, 2.0, "hi", None)],
                            offset=0.0, lead=0.0)
    text = out.read_text(encoding="utf-8")
  default = [l for l in text.splitlines() if l.startswith("Style: Default,")][0]
  assert export.ASS_UNSUNG_COLOUR in default
  assert "&H000000FF" not in default

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

def test_embed_subs_tags_the_language_from_the_sidecar_name(monkeypatch):
  seen = _mux_cmd(monkeypatch)
  with tempfile.TemporaryDirectory() as tmp:
    media = _tmp_media(tmp)
    srt = media.with_suffix(".ja.srt")
    srt.touch()
    export.embed_subs(media, srt)
  assert "language=jpn" in seen["cmd"]

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
  assert "\\kf" in line, line
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
  total = sum(int(n) for n in re.findall(r"\\kf(\d+)", line))
  assert total == 200, (total, line)

# --- gap 8: .ass style knobs ---

def test_ass_style_knobs_reach_the_style_lines():
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(
      _tmp_media(tmp), [Cue(1.0, 2.0, "hi", None)], offset=0.0, lead=0.0,
      style={"font": "Meiryo", "size": 40, "outline": 5, "box": True, "pos": 8})
    head = out.read_text(encoding="utf-8").split("[Events]")[0]
  assert "Style: Default,Meiryo,40," in head
  assert ",100,100,0,0,3,5,1,8,60,60," in head  # BorderStyle 3, top alignment

def test_ass_defaults_are_unchanged_without_style():
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), [Cue(1.0, 2.0, "hi", None)],
                            offset=0.0, lead=0.0)
    head = out.read_text(encoding="utf-8").split("[Events]")[0]
  assert f"Style: Default,{export.ASS_FONT},{export.ASS_FONT_SIZE}," in head
  assert f",100,100,0,0,1,{export.ASS_OUTLINE},{export.ASS_SHADOW},2,60,60," in head

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

# --- translation line ---

def _tr_cand(tlyric):
  from utasub.core.providers import Candidate
  return Candidate(title="晴る", album="", artist="ヨルシカ", source="NetEase",
                   lrc="[00:01.00]ゆめのつづき\n[00:03.00]あなたは夏の風", tlyric=tlyric)

def _ass_dialogue(cues, translations):
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), cues, offset=0.0, lead=0.0,
                            translations=translations)
    body = out.read_text(encoding="utf-8").split("[Events]")[1]
  return [ln for ln in body.splitlines() if ln.startswith("Dialogue:")]

def test_translation_adds_a_third_line_under_the_original():
  from utasub.core.providers import translation_lines
  cues = [Cue(0.0, 2.0, "ゆめのつづき", "lrc"), Cue(2.0, 4.0, "あなたは夏の風", "lrc")]
  cand = _tr_cand("[00:01.00]梦的延续\n[00:03.00]你似那夏日薰风")
  lines = _ass_dialogue(cues, translation_lines(cand))
  assert [ln.count("\\N") for ln in lines] == [2, 2]
  assert "梦的延续" in lines[0] and "你似那夏日薰风" in lines[1]
  assert "{\\rTranslation}" in lines[0]

def test_no_tlyric_leaves_two_lines():
  from utasub.core.providers import translation_lines
  cues = [Cue(0.0, 2.0, "ゆめのつづき", "lrc"), Cue(2.0, 4.0, "あなたは夏の風", "lrc")]
  lines = _ass_dialogue(cues, translation_lines(_tr_cand("")))
  assert [ln.count("\\N") for ln in lines] == [1, 1]

def test_translation_style_is_declared_in_the_header():
  with tempfile.TemporaryDirectory() as tmp:
    out = export.export_ass(_tmp_media(tmp), [Cue(0.0, 2.0, "ゆめ", "lrc")],
                            offset=0.0, lead=0.0)
    header = out.read_text(encoding="utf-8").split("[Events]")[0]
  assert "Style: Translation," in header
  assert export.ASS_TR_COLOUR in header

def test_translation_matches_a_shifted_timestamp_within_half_a_second():
  from utasub.core.providers import translation_lines
  cand = _tr_cand("[00:01.40]梦的延续\n[00:03.40]你似那夏日薰风")
  assert translation_lines(cand)["ゆめのつづき"] == "梦的延续"

def test_translation_falls_back_to_position_when_timestamps_are_unrelated():
  """A translation timed off another cut of the song lines up nowhere, so the
  nearest-timestamp pass finds nothing and order decides."""
  from utasub.core.providers import translation_lines
  cand = _tr_cand("[01:40.00]梦的延续\n[01:42.00]你似那夏日薰风")
  assert translation_lines(cand) == {"ゆめのつづき": "梦的延续",
                                     "あなたは夏の風": "你似那夏日薰风"}
