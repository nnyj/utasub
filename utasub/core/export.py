"""Write .srt (.ja/.zh/.ko.srt for CJK), optional romaji track, and styled .ass
(romaji over dimmed original)."""
import subprocess
from pathlib import Path

from .align import Cue
from .romanize import detect, is_cjk, romanize

# Export-time timing knobs. Both apply at write time only, so sessions and the
# timeline stay onset-true and either can change without re-aligning.
#   offset: whole track shifts, player convention (negative = earlier)
#   lead:   starts pulled earlier, ends kept, so a line may overlap the one before
SRT_OFFSET_S = -0.2
SRT_LEAD_S = 0.0


def apply_offset(segments, offset):
  """Shift every cue by `offset` seconds, durations unchanged (negative =
  earlier). Starts clamp at 0. Returns new list, element shapes preserved."""
  if not offset or not segments:
    return list(segments)
  out = []
  for seg in segments:
    start = seg[0] + offset
    end = seg[1] + offset
    if start < 0:  # clamp at the head, keep the cue visible
      end = max(end - start, 0.2)
      start = 0.0
    out.append(_retime(seg, start, end))
  return out


def apply_lead(segments, lead):
  """Start each cue `lead` seconds earlier, ends unchanged: line lingers and
  may overlap previous one. Starts clamp at 0."""
  if not lead or not segments:
    return list(segments)
  return [_retime(seg, max(0.0, seg[0] - lead), seg[1]) for seg in segments]


def _retime(seg, start, end):
  """Copy seg with new start/end, keeping its type (Cue or plain tuple)."""
  if isinstance(seg, Cue):
    return Cue(start, end, seg.text, seg.confidence)
  return (start, end) + tuple(seg[2:])


def _apply_timing(segments, offset, lead):
  """Apply write-time offset then lead, printing timing line when either set.
  Returns new list (unchanged when both zero)."""
  if offset or lead:
    print(f"  srt timing: offset {offset:+.2f}s, lead {lead:.2f}s")
    return apply_lead(apply_offset(segments, offset), lead)
  return segments


def srt_timestamp(seconds):
  ms = round(seconds * 1000)
  h, ms = divmod(ms, 3600000)
  m, ms = divmod(ms, 60000)
  s, ms = divmod(ms, 1000)
  return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(path, segments):
  """Write SRT file from [(start, end, text[, ...]), ...]. Returns path."""
  lines = []
  for i, seg in enumerate(segments, 1):
    start, end, text = seg[0], seg[1], seg[2]
    lines.append(f"{i}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}\n")
  Path(path).write_text("\n".join(lines), encoding="utf-8")
  return Path(path)


def make_romaji_segments(segments, locale=None):
  """Convert CJK text to romaji in segments. locale None detects once over all
  segments (song is single-language; per-line detect misreads kanji as zh)."""
  if locale is None:
    locale = detect("".join(seg[2] for seg in segments))
  out = []
  for seg in segments:
    start, end, text = seg[0], seg[1], seg[2]
    out.append((start, end, romanize(text, locale=locale)))
  return out


def srt_name(media_path, segments):
  """Pick .<locale>.srt for CJK text (ja/zh/ko), else plain .srt."""
  locale = detect("".join(seg[2] for seg in segments))
  suffix = f".{locale}.srt" if locale else ".srt"
  return Path(media_path).with_suffix(suffix)


def export_srt(media_path, segments, romaji=True, offset=None, lead=None):
  """Write main SRT + optional romaji track. Returns written paths.
  offset/lead default to SRT_OFFSET_S / SRT_LEAD_S."""
  offset = SRT_OFFSET_S if offset is None else offset
  lead = SRT_LEAD_S if lead is None else lead
  segments = _apply_timing(segments, offset, lead)
  out_path = srt_name(media_path, segments)
  written = [write_srt(out_path, segments)]
  print(f"  -> {out_path.name}")
  if romaji and is_cjk("".join(seg[2] for seg in segments)):
    rom_path = Path(media_path).with_suffix(".romaji.srt")
    write_srt(rom_path, make_romaji_segments(segments))
    written.append(rom_path)
    print(f"  -> {rom_path.name}")
  return written


# --- .ass (styled dual-line: romaji on top, original dimmed below) ---
# All knobs default here; override per call to restyle without touching cues.
ASS_FONT = "Arial"
ASS_FONT_SIZE = 72
ASS_DIM_SCALE = 0.72       # original-line size vs romaji, 0..1
ASS_DIM_ALPHA = "80"       # original-line transparency, ASS hex 00=opaque..FF=clear
ASS_FADE_MS = (120, 120)   # (fade-in, fade-out) per cue, ms
ASS_OUTLINE = 3
ASS_SHADOW = 1
ASS_MARGIN_V = 54          # gap from the bottom edge, PlayRes px
ASS_PLAY_RES = (1920, 1080)
ASS_MC_SCALE = 0.6         # MC (spoken) line size vs lyric lines
ASS_MC_COLOUR = "&H00B4B4B4"   # grey: patter reads as secondary to the song


def is_mc(seg):
  """True when cue carries the 'mc' confidence tag (spoken intermission)."""
  return len(seg) > 3 and seg[3] == "mc"


def ass_timestamp(seconds):
  """H:MM:SS.cs (centiseconds), ASS convention."""
  cs = round(seconds * 100)
  h, cs = divmod(cs, 360000)
  m, cs = divmod(cs, 6000)
  s, cs = divmod(cs, 100)
  return f"{h:d}:{m:02}:{s:02}.{cs:02}"


def _ass_header(font, font_size, outline, shadow, margin_v, play_res):
  w, h = play_res
  return (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    f"PlayResX: {w}\nPlayResY: {h}\n"
    "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"  # smart wrap keeps long romaji lines inside frame

    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
    "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
    "MarginR, MarginV, Encoding\n"
    f"Style: Default,{font},{font_size},&H00FFFFFF,&H000000FF,&H00000000,"
    f"&H00000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,60,60,{margin_v},1\n"
    f"Style: MC,{font},{max(1, round(font_size * ASS_MC_SCALE))},{ASS_MC_COLOUR},"
    f"&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},"
    f"2,60,60,{margin_v},1\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
    "Effect, Text\n")


def export_ass(media_path, cues, *, offset=None, lead=None):
  """Write <name>.ass. Japanese cues render romaji on top, original dimmed
  below; non-Japanese cues stay one line. offset/lead reuse SRT write-time
  knobs; style comes from module-level ASS_* constants.
  Returns written path."""
  offset = SRT_OFFSET_S if offset is None else offset
  lead = SRT_LEAD_S if lead is None else lead
  cues = _apply_timing(cues, offset, lead)
  # song-level locale, once, from sung lines only (patter must not sway it)
  locale = detect("".join(seg[2] for seg in cues if not is_mc(seg))) or None
  dim_size = max(1, round(ASS_FONT_SIZE * ASS_DIM_SCALE))
  fade_in, fade_out = ASS_FADE_MS
  fade = f"{{\\fad({fade_in},{fade_out})}}" if (fade_in or fade_out) else ""
  body = []
  for seg in cues:
    start, end, text = seg[0], seg[1], seg[2]
    if not text:
      continue
    style = "Default"
    if is_mc(seg):   # spoken: verbatim single line, grey MC style, no romaji
      style = "MC"
    else:
      rom = romanize(text, locale=locale) if is_cjk(text) else ""
      if rom and rom != text:
        text = f"{rom}\\N{{\\fs{dim_size}\\alpha&H{ASS_DIM_ALPHA}&}}{text}"
    body.append(f"Dialogue: 0,{ass_timestamp(start)},{ass_timestamp(end)},"
                f"{style},,0,0,0,,{fade}{text}")
  out_path = Path(media_path).with_suffix(".ass")
  header = _ass_header(ASS_FONT, ASS_FONT_SIZE, ASS_OUTLINE, ASS_SHADOW,
                       ASS_MARGIN_V, ASS_PLAY_RES)
  out_path.write_text(header + "\n".join(body) + "\n", encoding="utf-8")
  print(f"  -> {out_path.name}")
  return out_path


def embed_srt(media_path, srt_path, out_path=None):
  """Mux srt_path into media_path as soft subtitle track. Always writes an mkv
  (<name>.subbed.mkv) with stream copy: fast remux, not a re-encode.
  Language tag from .ja.srt name. Returns output Path, raises on ffmpeg error."""
  media_path = Path(media_path)
  srt_path = Path(srt_path)
  out_path = Path(out_path) if out_path else media_path.with_suffix(".subbed.mkv")
  lang = "jpn" if srt_path.name.lower().endswith(".ja.srt") else "und"
  cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(media_path), "-i", str(srt_path),
         "-map", "0", "-map", "1", "-c", "copy", "-c:s", "srt",
         "-metadata:s:s:0", f"language={lang}", str(out_path)]
  flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
  proc = subprocess.run(cmd, capture_output=True, creationflags=flags)
  if proc.returncode != 0:
    raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip() or "ffmpeg failed")
  return out_path
