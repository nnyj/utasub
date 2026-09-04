"""Write .srt (.ja/.zh/.ko.srt for CJK), optional romaji track, and styled .ass
(romaji over dimmed original)."""
import json
import subprocess
from dataclasses import replace
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
  return [_retime(seg, max(0.0, seg[0] - lead), seg[1],
                  span_shift=seg[0] - max(0.0, seg[0] - lead))
          for seg in segments]

def _retime(seg, start, end, span_shift=0.0):
  """Copy seg with new start/end, keeping its type (Cue or plain tuple).
  Karaoke spans are relative to the cue start, so they ride an offset for free
  and only a start-only move (lead) has to delay them by hand."""
  if not isinstance(seg, Cue):
    return (start, end) + tuple(seg[2:])
  out = replace(seg, start=start, end=end)
  if span_shift and out.token_spans:
    out.token_spans = [(c, a + span_shift, b + span_shift)
                       for c, a, b in out.token_spans]
  return out

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

def _reads_as_ja(text):
  """A han-only line detects as zh whether it is Japanese or Chinese. The
  Japanese romanizer marks the readings it does not have with `?`, which is
  what tells 漢字 (Kanji) from 我愛你 (Waga ai ?)."""
  rom = romanize(text, locale="ja")
  return bool(rom) and "?" not in rom and rom != text

def cue_locales(segments):
  """(locale per cue, majority locale). A concert is not one language: detected
  over the merged list, one Korean song renders as hangul and every Chinese line
  as literal `?`. Per cue instead, with the zh detection re-read as ja when the
  Japanese romanizer can read the line and the concert is mostly Japanese."""
  locs = [detect(seg[2]) for seg in segments]
  counts = {}
  for loc in locs:
    if loc:
      counts[loc] = counts.get(loc, 0) + 1
  major = max(counts, key=counts.get) if counts else ""
  if major == "ja":
    locs = [("ja" if loc == "zh" and _reads_as_ja(seg[2]) else loc)
            for seg, loc in zip(segments, locs)]
  return locs, major

def make_romaji_segments(segments, locale=None):
  """Convert CJK text to romaji in segments. locale None romanizes each cue in
  its own language."""
  locs = [locale] * len(segments) if locale else cue_locales(segments)[0]
  out = []
  for seg, loc in zip(segments, locs):
    start, end, text = seg[0], seg[1], seg[2]
    out.append((start, end, romanize(text, locale=loc)))
  return out

def srt_name(media_path, segments):
  """Pick .<locale>.srt for CJK text (ja/zh/ko), else plain .srt. The majority
  language names the track: a mixed concert has only one main SRT."""
  locale = cue_locales(segments)[1]
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
# SecondaryColour is the not-yet-sung half of a \kf fill. Grey, so a karaoke
# line reads as dim-then-white; lines without spans never use it.
ASS_UNSUNG_COLOUR = "&H00909090"
ASS_TR_COLOUR = "&H0080D8A0"   # pale green: translation reads apart from white sung line and grey MC
# Process-wide --translation: a third line under the original, when the chosen
# candidate carries one. Stored in the session, so re-export keeps it.
ASS_TRANSLATION = False
# Process-wide style overrides from the CLI flags: font, size, outline, box,
# pos. Empty = the constants above. Stored in the session, so a re-export
# reproduces the header without the flags.
ASS_STYLE = {}

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

def _kf_line(rom, spans, cue_len):
  r"""Romaji line as {\kf} runs from CTC token spans, or None when the spans do
  not line up with this romaji (a different locale pinned at export, or a hand
  text edit since the align).

  Matching is by character, not by count: the aligner drops whitespace and any
  char outside the MMS dictionary, so spans are a subsequence of `rom` and the
  untimed chars in between ride on the run before them. Durations are
  centiseconds and sum to the cue length, so the fill finishes exactly as the
  line leaves the screen."""
  runs, k, pending = [], 0, ""
  for ch in rom:
    if k < len(spans) and ch.lower() == spans[k][0]:
      runs.append([spans[k][1], pending + ch])
      pending = ""
      k += 1
    elif runs:
      runs[-1][1] += ch
    else:
      pending += ch
  if k != len(spans) or not runs:
    return None
  # each run holds the screen until the next one starts, so the inter-token gaps
  # of a held note are swept too. Boundaries round to whole centiseconds and the
  # durations are their differences, so the runs tile the cue exactly instead of
  # drifting a rounding step per run. Clamped into the cue: the envelope can trim
  # an end back inside the spans, an un-clamped tail overruns the line it paints.
  cs = [round(min(cue_len, max(0.0, r[0])) * 100) for r in runs] + [round(cue_len * 100)]
  out = f"{{\\kf{cs[0]}}}" if cs[0] else ""
  for i, (_start, text) in enumerate(runs):
    out += f"{{\\kf{max(0, cs[i + 1] - cs[i])}}}{text}"
  return out

def ass_style(style=None):
  """Effective .ass style: defaults, then ASS_STYLE, then `style`.
  Keys: font, size, outline, box (BorderStyle 3 opaque box), pos (2 bottom /
  8 top)."""
  out = {"font": ASS_FONT, "size": ASS_FONT_SIZE, "outline": ASS_OUTLINE,
         "box": False, "pos": 2}
  for src in (ASS_STYLE, style):
    out.update({k: v for k, v in (src or {}).items() if v is not None})
  return out

def _ass_header(font, font_size, outline, shadow, margin_v, play_res,
                border=1, align=2):
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
    f"Style: Default,{font},{font_size},&H00FFFFFF,{ASS_UNSUNG_COLOUR},&H00000000,"
    f"&H00000000,0,0,0,0,100,100,0,0,{border},{outline},{shadow},{align},60,60,"
    f"{margin_v},1\n"
    f"Style: MC,{font},{max(1, round(font_size * ASS_MC_SCALE))},{ASS_MC_COLOUR},"
    f"&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,{border},{outline},"
    f"{shadow},{align},60,60,{margin_v},1\n"
    f"Style: Translation,{font},{max(1, round(font_size * ASS_DIM_SCALE))},"
    f"{ASS_TR_COLOUR},&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,"
    f"{border},{outline},{shadow},{align},60,60,{margin_v},1\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
    "Effect, Text\n")

def export_ass(media_path, cues, *, offset=None, lead=None, style=None,
               translations=None):
  """Write <name>.ass. Japanese cues render romaji on top, original dimmed
  below; non-Japanese cues stay one line. offset/lead reuse SRT write-time
  knobs; style overrides the module-level ASS_* constants (see `ass_style`).
  translations: {original text: translated text} from `providers.translation_lines`,
  adding a third line in the Translation style. Returns written path."""
  offset = SRT_OFFSET_S if offset is None else offset
  lead = SRT_LEAD_S if lead is None else lead
  st = ass_style(style)
  cues = _apply_timing(cues, offset, lead)
  # per cue: a concert can hold a Japanese, a Korean and a Chinese song, and one
  # locale over the merged list renders two of the three as their own script
  locales = cue_locales(cues)[0]
  dim_size = max(1, round(st["size"] * ASS_DIM_SCALE))
  fade_in, fade_out = ASS_FADE_MS
  fade = f"{{\\fad({fade_in},{fade_out})}}" if (fade_in or fade_out) else ""
  body = []
  for seg, locale in zip(cues, locales):
    start, end, text = seg[0], seg[1], seg[2]
    if not text:
      continue
    style = "Default"
    rom = romanize(text, locale=locale or None) if is_cjk(text) else ""
    if is_mc(seg):
      # patter is secondary, so it gets the romaji alone in the grey MC style,
      # no dimmed original underneath
      style = "MC"
      text = rom or text
    else:
      # the top line is what the viewer sings along to, romaji for a CJK line
      # and the line itself for an English one, so that is where the fill goes
      dual = bool(rom and rom != text)
      sung = rom if dual else text
      spans = getattr(seg, "token_spans", None)
      sung = (_kf_line(sung, spans, end - start) if spans else None) or sung
      text = (f"{sung}\\N{{\\fs{dim_size}\\alpha&H{ASS_DIM_ALPHA}&\\k0}}{text}"
              if dual else sung)
      # \r switches the rest of the line to the named style, so the colour and
      # the dim size both come from the header rather than more inline tags
      tr = (translations or {}).get(seg[2])
      if tr:
        text += f"\\N{{\\rTranslation}}{tr}"
    body.append(f"Dialogue: 0,{ass_timestamp(start)},{ass_timestamp(end)},"
                f"{style},,0,0,0,,{fade}{text}")
  out_path = Path(media_path).with_suffix(".ass")
  header = _ass_header(st["font"], st["size"], st["outline"], ASS_SHADOW,
                       ASS_MARGIN_V, ASS_PLAY_RES,
                       border=3 if st["box"] else 1, align=int(st["pos"]))
  out_path.write_text(header + "\n".join(body) + "\n", encoding="utf-8")
  print(f"  -> {out_path.name}")
  return out_path

# --- .lrc (corrected timings back into the curated library) ---

def lrc_timestamp(seconds):
  """mm:ss.xx, LRC convention. Minutes are not wrapped at 60."""
  cs = round(max(0.0, seconds) * 100)
  m, cs = divmod(cs, 6000)
  s, cs = divmod(cs, 100)
  return f"{m:02}:{s:02}.{cs:02}"

def _region_bounds(region):
  """(start, end) from a Region or a (start, end) pair, whole track for None."""
  if region is None:
    return 0.0, float("inf")
  if hasattr(region, "start"):
    return region.start, region.end
  return region[0], region[1]

def lrc_text(cues, region=None, candidate=None):
  """LRC body: `[mm:ss.xx] text` per cue, rebased to the region start so the
  file times the song, not its place in the concert. Cues outside the region
  are dropped; `[ti:]/[ar:]` come from the chosen candidate."""
  start, end = _region_bounds(region)
  lines = []
  for tag, value in (("ti", getattr(candidate, "title", "")),
                     ("ar", getattr(candidate, "artist", ""))):
    if value:
      lines.append(f"[{tag}:{value}]")
  for seg in cues:
    if seg[0] < start or seg[0] > end or not seg[2]:
      continue
    lines.append(f"[{lrc_timestamp(seg[0] - start)}] {seg[2]}")
  return "\n".join(lines) + "\n"

def export_lrc(media_path, cues, region=None, candidate=None, out_dir=None):
  """Write the region's cues as an .lrc. Named `<artist> - <title>.lrc` in
  out_dir (the curated library) when both are known, else `<media>.lrc` beside
  the media. Returns written path."""
  name = None
  if out_dir:
    title = getattr(candidate, "title", "") or ""
    artist = getattr(candidate, "artist", "") or ""
    if title:
      name = f"{artist} - {title}" if artist else title
  if name:
    from .providers import _sanitize_filename
    out_path = Path(out_dir) / f"{_sanitize_filename(name)}.lrc"
  else:
    out_path = Path(media_path).with_suffix(".lrc")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(lrc_text(cues, region, candidate), encoding="utf-8")
  print(f"  -> {out_path.name}")
  return out_path

# --- mux (soft subtitle tracks) ---

def _sub_codec(path):
  """mkv subtitle codec for a sidecar: ass keeps the styling, srt is plain."""
  return "ass" if path.suffix.lower() in (".ass", ".ssa") else "srt"

SUB_LANGS = {"ja": "jpn", "zh": "chi", "ko": "kor"}  # mkv wants ISO 639-2

def _sub_lang(path, siblings=()):
  """Language tag from the sidecar name (.ja.srt, .zh.ass), else the tag of a
  sibling sidecar in the same mux (a bare <name>.ass rides its .ja.srt), else und."""
  def named(p):
    parts = p.name.lower().split(".")
    return SUB_LANGS.get(parts[-2]) if len(parts) > 2 else None

  own = named(path)
  if own:
    return own
  return next((named(p) for p in siblings or () if named(p)), "und")

def count_subtitle_streams(media_path):
  """Subtitle streams already in the source. `-map 0` copies them first, so the
  new tracks are output subtitles `n..`, and every -c:s / -metadata / -disposition
  index has to count from n or it lands on a pre-existing track. 0 on any probe
  failure."""
  cmd = ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index", "-of", "json", str(media_path)]
  flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
  try:
    proc = subprocess.run(cmd, capture_output=True, creationflags=flags)
    if proc.returncode != 0:
      return 0
    data = json.loads((proc.stdout or b"").decode("utf-8", "replace"))
    return len(data.get("streams") or [])
  except Exception:
    return 0

def embed_subs(media_path, sub_paths, out_path=None):
  """Mux every sidecar in `sub_paths` into media_path as soft subtitle tracks.
  Always writes an mkv (<name>.subbed.mkv) with stream copy: fast remux, not a
  re-encode. The .ass tracks go first and the first one is flagged default, so
  a player picks the styled dual-line track and leaves the .srt as the
  fallback. Returns output Path, raises on ffmpeg error."""
  media_path = Path(media_path)
  subs = [Path(p) for p in ([sub_paths] if isinstance(sub_paths, (str, Path))
                            else sub_paths)]
  subs = [p for p in subs if p.exists()]
  if not subs:
    raise RuntimeError("no subtitle file to mux")
  subs.sort(key=lambda p: _sub_codec(p) != "ass")
  out_path = Path(out_path) if out_path else media_path.with_suffix(".subbed.mkv")

  cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(media_path)]
  for p in subs:
    cmd += ["-i", str(p)]
  cmd += ["-map", "0"]
  for i in range(len(subs)):
    cmd += ["-map", str(i + 1)]
  cmd += ["-c", "copy"]
  base = count_subtitle_streams(media_path)
  for j in range(base):  # only one default track, and it is ours
    cmd += [f"-disposition:s:{j}", "0"]
  for i, p in enumerate(subs):
    j = base + i
    cmd += [f"-c:s:{j}", _sub_codec(p),
            f"-metadata:s:s:{j}", f"language={_sub_lang(p, subs)}",
            f"-metadata:s:s:{j}", f"title={p.suffix.lstrip('.').upper()}",
            f"-disposition:s:{j}", "default" if i == 0 else "0"]
  cmd.append(str(out_path))
  flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
  proc = subprocess.run(cmd, capture_output=True, creationflags=flags)
  if proc.returncode != 0:
    raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip() or "ffmpeg failed")
  return out_path
