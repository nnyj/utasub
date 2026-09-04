"""Entry point. Flags: --gui/--no-gui, --multi-song, --asr, --asr-fill,
--providers, --fa, --romaji, --mc/--no-mc, --fresh.
Headless best-effort chain per README; drag-drop via utasub.bat."""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from .core.align import SIMILARITY_THRESHOLD
from .core.multi_song import (_multi_song_headless, _multi_song_prepare,
                              _load_audio_once, detect_song_regions,
                              export_and_save, media_duration_s)

MEDIA_EXTS = (".mkv", ".mp4", ".webm", ".flac", ".m4a", ".mp3", ".wav")

# What this tool writes beside a source: the mux output, the separation stems,
# the session sidecar. Ingesting one re-runs stems + ASR on a file already done
# and grows the name every pass (.subbed.subbed.mkv). Named targets are still
# honoured, this only filters directory and glob expansion.
DERIVED_STEMS = (".subbed", ".vocals", ".instrumental", ".stem", ".utasub")

# ━━━━━━ targets ━━━━━━

def is_derived(path):
  """True for a file this tool produced, by its second-level suffix."""
  return Path(Path(path).stem).suffix.lower() in DERIVED_STEMS

def expand_targets(targets, recursive=False):
  """Media files to process. A directory expands to its media files, a literal
  glob is expanded here (cmd and Explorer never do), anything else passes
  through. Own outputs skipped, order preserved, duplicates dropped."""
  out = []
  for target in targets:
    p = Path(target)
    if p.is_dir():
      found = p.rglob("*") if recursive else p.glob("*")
      out.extend(sorted(f for f in found
                        if f.is_file() and f.suffix.lower() in MEDIA_EXTS
                        and not is_derived(f)))
    elif any(ch in target for ch in "*?["):
      out.extend(sorted(f for f in p.parent.glob(p.name) if not is_derived(f)))
    else:
      out.append(p)
  seen = set()
  unique = []
  for p in out:
    key = os.path.normcase(os.path.abspath(p))
    if key not in seen:
      seen.add(key)
      unique.append(p)
  return unique

def _has_display():
  """True when a GUI display is likely available."""
  if sys.platform == "win32":
    return True
  return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

# ━━━━━━ single-song headless align ━━━━━━

def _align_and_export(path, best, segments, candidates, romaji, opts, edits=None, audio=None):
  """Parse, align via placement chain, export, save session. audio is decoded
  here when absent. Returns (status, strategy). opts: place.AlignOpts (use_fa)."""
  from lyrickit import classify_lines, filter_credit_lines
  from .core.place import run_chain

  lines = best.lines
  # pin locale once from whole lyric: per-line detect misreads kanji-only
  # Japanese as zh; songs are single-language
  from .core.romanize import detect, set_default_locale
  set_default_locale(detect("\n".join(t for _, t in lines)))

  if not lines:
    print("  LRC parse empty, keeping pure ASR")
    export_and_save(path, segments, romaji, edits=edits, candidates=candidates)
    return "pure-ASR", "parse_empty"

  verdicts = classify_lines(lines)
  print(f"  credits: {verdicts.count('credit')} lines struck")
  credit_toggles = {str(i): v for i, v in enumerate(verdicts) if v == "credit"}
  clean_lines = filter_credit_lines(lines, verdicts)
  if not clean_lines:
    print("  all lines struck as credits, keeping pure ASR")
    export_and_save(path, segments, romaji, edits=edits, candidates=candidates,
                    credit_toggles=credit_toggles)
    return "pure-ASR", "all_credits"

  if audio is None:
    audio, audio_path = _load_audio_once(path)
    if opts.use_fa and str(audio_path) != str(path):
      print(f"  fa: using vocals stem {Path(audio_path).name}")

  aligned, song_span, meta = run_chain(clean_lines, segments, audio, opts)
  strategy = meta.get("strategy", "none")
  print(f"  placement: {strategy}")
  chosen_idx = next((i for i, c in enumerate(candidates) if c is best), 0)
  export_and_save(path, aligned, romaji, edits=edits, candidates=candidates,
                  chosen_index=chosen_idx, song_span=song_span,
                  credit_toggles=credit_toggles, placement=meta)
  return ("picked" if song_span else "pure-ASR"), strategy

def load_setlist_file(path):
  """Read a manual setlist text file, parse it into ordered titles."""
  from .core.setlist import parse_pasted
  return parse_pasted(Path(path).read_text(encoding="utf-8", errors="replace"))

# ━━━━━━ main ━━━━━━

def main():
  # pythonw: sys.stdout/stderr may be None
  if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  if sys.stderr is not None:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
  parser = argparse.ArgumentParser(description="Stamp official lyrics onto performance videos")
  parser.add_argument("targets", nargs="*",
                      help="media files, directories or globs; omit to open GUI empty")
  gui_default = _has_display()
  parser.add_argument("--gui", action=argparse.BooleanOptionalAction,
                      default=gui_default,
                      help="launch picker GUI (default when display available)")
  parser.add_argument("--recursive", action="store_true", default=False,
                      help="descend into subdirectories when a target is a directory")
  parser.add_argument("--multi-song", action="store_true", default=False,
                      help="multi-song region detection and per-region lyric search")
  parser.add_argument("--providers", default="NetEase",
                      help="comma-separated lyric providers")
  try:
    from .core.ctc_align import ctc_available
    from .core.fa import fa_available
    fa_default = ctc_available() or fa_available()
  except Exception:
    fa_default = False
  parser.add_argument("--fa", action=argparse.BooleanOptionalAction, default=fa_default,
                      help="forced alignment (default ON when torchaudio or qwen_asr available)")
  parser.add_argument("--asr", action=argparse.BooleanOptionalAction, default=True,
                      help="transcribe in-process when the file has no ASR block "
                           "(default ON; utasub-asr still does batch pre-transcription)")
  parser.add_argument("--asr-fill", action=argparse.BooleanOptionalAction, default=True,
                      help="keep transcript outside song spans as cues, junk filtered "
                           "(default ON; off exports lyrics and MC only)")
  parser.add_argument("--romaji", action=argparse.BooleanOptionalAction, default=True,
                      help="write romaji .srt track for Japanese")
  parser.add_argument("--lang", choices=("auto", "jp", "cn", "kr"), default="auto",
                      help="lyric language for romanization, auto detects per song (default)")
  parser.add_argument("--setlist", action=argparse.BooleanOptionalAction, default=True,
                      help="setlist discovery for unassigned multi-song regions (default ON in --multi-song)")
  parser.add_argument("--setlist-file", default=None, metavar="PATH",
                      help="text file with one song title per line (numbered/bulleted "
                           "lists tolerated), used instead of setlist discovery")
  parser.add_argument("--mc", action=argparse.BooleanOptionalAction, default=True,
                      help="detect spoken intermissions: kept out of song matching, "
                           "exported as their own grey romaji cues (default ON)")
  parser.add_argument("--timeline", action="store_true", default=False,
                      help="open MainWindow timeline directly from a saved session (no re-align)")
  parser.add_argument("--fresh", action="store_true", default=False,
                      help="bypass fetch cache")
  from .core import export as export_mod
  parser.add_argument("--offset", type=float, default=export_mod.SRT_OFFSET_S,
                      help="SRT offset in seconds, negative = subtitles appear "
                           f"earlier (default {export_mod.SRT_OFFSET_S})")
  parser.add_argument("--lead", type=float, default=export_mod.SRT_LEAD_S,
                      help="extra head start in seconds: cues begin earlier, ends "
                           f"kept, lines may overlap (default {export_mod.SRT_LEAD_S})")
  parser.add_argument("--ass-font", default=export_mod.ASS_FONT,
                      help=f".ass font name (default {export_mod.ASS_FONT})")
  parser.add_argument("--ass-size", type=int, default=export_mod.ASS_FONT_SIZE,
                      help=f".ass font size in PlayRes px (default {export_mod.ASS_FONT_SIZE})")
  parser.add_argument("--ass-outline", type=float, default=export_mod.ASS_OUTLINE,
                      help=f".ass outline thickness (default {export_mod.ASS_OUTLINE})")
  parser.add_argument("--ass-box", action="store_true", default=False,
                      help="opaque box behind the text (BorderStyle 3), readable "
                           "over bright stage video")
  parser.add_argument("--ass-pos", type=int, choices=(2, 8), default=2,
                      help="subtitle position: 2 bottom (default), 8 top")
  parser.add_argument("--translation", action=argparse.BooleanOptionalAction,
                      default=False,
                      help="third .ass line with the candidate's translated "
                           "lyrics, when the provider carries one (default OFF)")
  args = parser.parse_args()

  export_mod.SRT_OFFSET_S = args.offset  # process-wide export defaults
  export_mod.SRT_LEAD_S = args.lead
  export_mod.ASS_STYLE = {"font": args.ass_font, "size": args.ass_size,
                          "outline": args.ass_outline, "box": args.ass_box,
                          "pos": args.ass_pos}
  export_mod.ASS_TRANSLATION = args.translation
  from .core.romanize import set_lang_override
  set_lang_override(args.lang)  # beats per-song detection everywhere, incl. GUI menu
  providers = [p.strip() for p in args.providers.split(",") if p.strip()]
  setlist_titles = load_setlist_file(args.setlist_file) if args.setlist_file else None
  if args.setlist_file:
    print(f"  setlist file: {len(setlist_titles)} titles from {args.setlist_file}")
  from .core.place import AlignOpts
  opts = AlignOpts(use_fa=args.fa)

  # no targets + GUI: open empty MainWindow
  if not args.targets and args.gui:
    from .gui.main_window import open_main_window
    open_main_window(providers=providers, fresh=args.fresh,
                     romaji=args.romaji, opts=opts)
    return

  if not args.targets:
    parser.error("no targets specified (use --gui to open without files)")

  from .core import asr_gen, session
  from .core.asr import load_asr
  from .core.providers import fetch_candidates, media_tags
  from .core.align import score_candidates
  from .core.romanize import set_default_locale

  files = expand_targets(args.targets, recursive=args.recursive)
  if not files:
    parser.error("no media files in the given targets")
  rows = []

  for path in files:
    print(f"=== {path.name}")
    set_default_locale("")  # no locale leak from previous file
    row = {"file": path.name, "regions": "-", "assigned": "-",
           "strategy": "-", "result": "skipped"}
    rows.append(row)

    # --timeline always opens the saved session in the GUI, never the align path
    if args.timeline:
      from .gui.main_window import open_main_window
      open_main_window(session_only=str(path), providers=providers,
                       fresh=args.fresh, romaji=args.romaji, opts=opts)
      row["result"] = "timeline"
      continue

    # manual edits outlive a re-align, --fresh included: they are the one thing
    # in the session the pipeline can never reproduce
    sess = session.load(path)
    edits = (sess or {}).get("manual_edits") or {}
    n_edits = len(edits.get("timings") or []) + len(edits.get("deleted") or [])
    if n_edits:
      print(f"  keeping {n_edits} manual edit(s)")

    # re-export shortcut is headless-only: GUI always opens picker to refine
    # (fetch cache makes reopening free)
    if sess and sess.get("cues") and not args.gui and not args.fresh:
      session.reexport(path, sess)
      print(f"  result: re-export ({len(sess['cues'])} cues)")
      row["strategy"], row["result"] = "re-export", f"{len(sess['cues'])} cues"
      continue

    result = load_asr(path)
    if result is None and args.asr and asr_gen.asr_available():
      print("  no ASR block, transcribing (minutes, GPU)...")
      asr_gen.transcribe(path)
      result = load_asr(path)
    if result is None:
      print("  skip (no ASR - run utasub-asr first)")
      row["result"] = "no ASR"
      continue
    segments, _ = result

    # --- multi-song path ---
    if args.multi_song:
      audio, audio_path = _load_audio_once(path)
      if audio is not None and str(audio_path) != str(path):
        print(f"  audio: {Path(audio_path).name}")
      if not args.gui:
        regions, metas = _multi_song_headless(
          path, segments, audio, providers, args.fresh, args.romaji, opts,
          use_setlist=args.setlist, mc=args.mc, setlist_titles=setlist_titles,
          asr_fill=args.asr_fill, edits=edits)
        row["regions"] = len(regions)
        row["assigned"] = sum(1 for m in metas if m.get("title"))
        # strategy column: distinct strategies with counts, busiest first
        counts = Counter(m.get("strategy", "none") for m in metas)
        row["strategy"] = ", ".join(f"{s} x{n}" if n > 1 else s
                                    for s, n in counts.most_common()) or "-"
        row["result"] = "multi-song" if regions else "pure-ASR"
        continue

      # prepare: detect + search + score (same as headless)
      prepared = _multi_song_prepare(path, segments, audio, providers,
                                     args.fresh, use_setlist=args.setlist,
                                     mc=args.mc, setlist_titles=setlist_titles)
      if prepared is None:
        print("  no regions found, keeping pure ASR")
        export_and_save(path, segments, args.romaji, edits=edits)
        row["result"] = "pure-ASR"
        continue
      regions, per_region_scored, assignments, notes, _mc_flags = prepared
      tags = media_tags(path)
      # MainWindow: picker + timeline + finalize all inside GUI
      from .gui.main_window import open_main_window
      exported = open_main_window(
        media_path=str(path), tags=tags, segments=segments,
        media_dur_ms=tags.get("_duration_ms", 0), providers=providers,
        fresh=args.fresh, regions=regions, region_scored=per_region_scored,
        region_assignments=assignments, audio=audio,
        romaji=args.romaji, opts=opts)
      row["regions"] = len(regions)
      row["assigned"] = sum(1 for a in assignments if a is not None)
      if exported:
        print("  result: multi-song (GUI)")
        row["result"] = "multi-song (GUI)"
      else:
        print("  closed without export, keeping pure ASR")
        export_and_save(path, segments, args.romaji, edits=edits)
        row["result"] = "pure-ASR"
      continue

    # fetch candidates (single ffprobe via media_tags)
    tags = media_tags(path)
    media_dur_ms = tags.get("_duration_ms", 0)
    candidates = fetch_candidates(path, providers, args.fresh, tags=tags)
    scored = score_candidates(candidates, segments, media_dur_ms)
    row["regions"] = 1

    # --- GUI path: whole song = one full-span region, drive MainWindow ---
    if args.gui:
      audio, audio_path = _load_audio_once(path)
      if audio is not None and str(audio_path) != str(path):
        print(f"  audio: {Path(audio_path).name}")
      from .core.regions import Region
      duration_s = media_duration_s(audio, segments, media_dur_ms)
      from .gui.main_window import open_main_window
      exported = open_main_window(
        media_path=str(path), tags=tags, segments=segments,
        media_dur_ms=media_dur_ms, providers=providers, fresh=args.fresh,
        regions=[Region(0.0, duration_s)], region_scored=[scored],
        region_assignments=[None], audio=audio,
        romaji=args.romaji, opts=opts)
      if exported:
        print("  result: single-song (GUI)")
        row["assigned"], row["result"] = 1, "single-song (GUI)"
      else:
        print("  closed without export, keeping pure ASR")
        export_and_save(path, segments, args.romaji, edits=edits,
                        candidates=candidates)
        row["result"] = "pure-ASR"
      continue

    # auto-suggest: cheap region detection (headless hint only)
    audio = None
    if len(segments) > 50:
      audio, _ = _load_audio_once(path)
      n_regions = len(detect_song_regions(segments, audio))
      if n_regions >= 2:
        print(f"  hint: {n_regions} song regions detected, --multi-song splits them")

    # --- headless path ---
    if not scored:
      msg = "no lyrics found" if not candidates else "no lyrics with text found"
      print(f"  {msg}, keeping pure ASR")
      export_and_save(path, segments, args.romaji, edits=edits,
                      candidates=candidates)
      print("  result: pure-ASR")
      row["result"] = "pure-ASR"
      continue

    best_score, best = scored[0]
    print(f"  top candidate: {best.title} [{best.source}] sim={best_score:.2f}")

    if best_score < SIMILARITY_THRESHOLD:
      print(f"  below threshold ({SIMILARITY_THRESHOLD}), keeping pure ASR")
      export_and_save(path, segments, args.romaji, edits=edits,
                      candidates=candidates)
      print("  result: pure-ASR")
      row["result"] = "pure-ASR"
      continue

    status, strategy = _align_and_export(path, best, segments, candidates,
                                         args.romaji, opts, edits=edits, audio=audio)
    print(f"  result: {status}")
    row["assigned"], row["strategy"], row["result"] = (1 if status == "picked" else 0), strategy, status

  # one-screen triage table for a batch run
  if len(rows) > 1:
    columns = ("file", "regions", "assigned", "strategy", "result")
    table = [columns] + [tuple(str(r[c]) for c in columns) for r in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(columns))]
    print("\n=== summary")
    for i, row in enumerate(table):
      print("  " + " | ".join(c.ljust(w) for c, w in zip(row, widths)))
      if i == 0:
        print("  " + "-+-".join("-" * w for w in widths))

if __name__ == "__main__":
  main()
