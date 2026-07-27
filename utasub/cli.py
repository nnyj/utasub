"""Entry point. Flags: --gui/--no-gui, --multi-song, --providers, --fa, --romaji,
--mc/--no-mc, --fresh.
Headless best-effort chain per README; drag-drop via utasub.bat."""
import argparse
import os
import re
import sys
from pathlib import Path

from .core.align import SIMILARITY_THRESHOLD

# setlist-matched pairs accept lower sim (order constraint compensates)
SETLIST_SIM_THRESHOLD = 0.30
# artist-pool greedy fallback: high enough to reject noise
POOL_SIM_THRESHOLD = 0.55


def _dedupe_by_title(scored):
  """Keep first (score, Candidate) per (title, artist), preserving order."""
  seen = set()
  out = []
  for s, c in scored:
    key = (c.title, c.artist)
    if key not in seen:
      seen.add(key)
      out.append((s, c))
  return out


def _has_display():
  """True when a GUI display is likely available."""
  if sys.platform == "win32":
    return True
  return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _align_and_export(path, best, segments, candidates, romaji, opts,
                      picker_lines_verdicts=None):
  """Parse, align via placement chain, export, save session. Returns status string.
  opts: place.AlignOpts (use_fa).
  picker_lines_verdicts: (lines, verdicts) from picker, skips re-parse + re-classify."""
  from lyrickit import classify_lines, filter_credit_lines
  from .core.align import load_audio_16k, find_vocals
  from .core.place import run_chain
  from .core.export import export_srt
  from .core import session

  if picker_lines_verdicts is not None:
    lines, verdicts = picker_lines_verdicts
    if lines and verdicts and len(verdicts) == len(lines):
      print(f"  credits: {verdicts.count('credit')} lines struck (picker)")
    else:
      lines = best.lines
      verdicts = None
  else:
    lines = best.lines
    verdicts = None

  # pin locale once from whole lyric: per-line detect misreads kanji-only
  # Japanese as zh; songs are single-language
  from .core.romanize import detect, set_default_locale
  set_default_locale(detect("\n".join(t for _, t in lines)))

  if not lines:
    print(f"  LRC parse empty, keeping pure ASR")
    export_srt(path, segments, romaji=romaji)
    session.save(path, candidates=candidates, cues=segments, romaji=romaji)
    return "pure-ASR"

  if verdicts is None:
    verdicts = classify_lines(lines)
    print(f"  credits: {verdicts.count('credit')} lines struck")
  credit_toggles = {str(i): v for i, v in enumerate(verdicts) if v == "credit"}
  clean_lines = filter_credit_lines(lines, verdicts)
  if not clean_lines:
    print(f"  all lines struck as credits, keeping pure ASR")
    export_srt(path, segments, romaji=romaji)
    session.save(path, candidates=candidates, cues=segments, romaji=romaji,
                 credit_toggles=credit_toggles)
    return "pure-ASR"

  audio = None
  audio_path = find_vocals(path) or path
  if opts.use_fa and str(audio_path) != str(path):
    print(f"  fa: using vocals stem {Path(audio_path).name}")
  try:
    audio = load_audio_16k(audio_path)
  except Exception as e:
    print(f"  audio load failed ({e}), aligning without envelope")

  aligned, song_span, meta = run_chain(clean_lines, segments, audio, opts)
  print(f"  placement: {meta.get('strategy', 'none')}")
  export_srt(path, aligned, romaji=romaji)

  chosen_idx = next((i for i, c in enumerate(candidates) if c is best), 0)
  session.save(path, candidates=candidates, chosen_index=chosen_idx,
               cues=aligned, song_span=song_span, romaji=romaji,
               credit_toggles=credit_toggles, placement=meta)
  return "picked" if song_span else "pure-ASR"


# --- multi-song flow ---

def _load_audio_once(path):
  """Load audio from vocals stem or media, return (audio, audio_path)."""
  from .core.align import load_audio_16k, find_vocals
  audio_path = find_vocals(path) or path
  try:
    audio = load_audio_16k(audio_path)
    return audio, audio_path
  except Exception as e:
    print(f"  audio load failed ({e})")
    return None, audio_path


def _setlist_dp_assign(regions, tracklist, segments, providers, artist_hint,
                       fresh, notes, mc_exclude=None,
                       source_label="Setlist:Wikipedia"):
  """Order-aware monotonic DP: match time-ordered regions to setlist tracks.
  Fetches lyrics per title, builds R×T score matrix, finds max-score monotonic
  matching allowing skips on both sides.
  Returns (assignments, per_region_scored)."""
  from .core.regions import region_segments
  from .core.providers import fetch_from_providers
  from .core.align import score_candidates

  R = len(regions)
  T = len(tracklist)

  track_candidates = []  # list of list[Candidate]
  fetch_failures = 0
  for order, title in tracklist:
    query = f"{title} {artist_hint}" if artist_hint else title
    cands = fetch_from_providers(query, providers, fresh)
    has_lrc = any(c.lrc for c in cands)
    if not has_lrc:
      fetch_failures += 1
    track_candidates.append(cands)
  print(f"  setlist: fetched {T} tracks, {fetch_failures} without lyrics")

  score_mat = [[None] * T for _ in range(R)]  # (score, Candidate) or None
  per_region_scored = [[] for _ in range(R)]
  for i, region in enumerate(regions):
    rsegs = region_segments(segments, region, exclude=mc_exclude)
    media_dur_ms = int(region.duration * 1000)
    all_scored = []
    for j, cands in enumerate(track_candidates):
      if not cands:
        continue
      scored = score_candidates(cands, rsegs, media_dur_ms)
      if scored:
        score_mat[i][j] = scored[0]  # best candidate for this track
        all_scored.extend(scored)
    all_scored.sort(key=lambda x: -x[0])
    per_region_scored[i] = _dedupe_by_title(all_scored)

  # DP: dp[i][j] = best total score matching regions[:i] × tracks[:j]
  dp = [[0.0] * (T + 1) for _ in range(R + 1)]
  choice = [[0] * (T + 1) for _ in range(R + 1)]
  # choice: 0=skip-region, 1=skip-track, 2=match
  for i in range(1, R + 1):
    for j in range(1, T + 1):
      dp[i][j] = dp[i - 1][j]
      choice[i][j] = 0
      if dp[i][j - 1] > dp[i][j]:
        dp[i][j] = dp[i][j - 1]
        choice[i][j] = 1
      # match (i-1, j-1) if above threshold
      pair = score_mat[i - 1][j - 1]
      if pair is not None and pair[0] >= SETLIST_SIM_THRESHOLD:
        val = dp[i - 1][j - 1] + pair[0]
        if val > dp[i][j]:
          dp[i][j] = val
          choice[i][j] = 2

  assignments = [None] * R
  i, j = R, T
  while i > 0 and j > 0:
    if choice[i][j] == 2:
      pair = score_mat[i - 1][j - 1]
      pair_cand = pair[1]
      if not str(pair_cand.source).startswith("Setlist:"):
        pair_cand.source = source_label
      assignments[i - 1] = pair
      i -= 1
      j -= 1
    elif choice[i][j] == 1:
      j -= 1
    else:
      i -= 1

  matched = sum(1 for a in assignments if a is not None)
  msg = f"setlist DP: {matched}/{R} regions matched ({fetch_failures} tracks without lyrics)"
  print(f"  {msg}")
  notes.append(msg)
  return assignments, per_region_scored


def load_setlist_file(path):
  """Read a manual setlist text file, parse it into ordered titles."""
  from .core.setlist import parse_pasted
  return parse_pasted(Path(path).read_text(encoding="utf-8", errors="replace"))


def _multi_song_prepare(path, segments, audio, providers, fresh,
                        use_setlist=True, mc=True, setlist_titles=None):
  """Detect regions, discover setlist (primary) or artist-pool greedy (fallback).
  Returns (regions, per_region_scored, assignments, discovery_notes) or None.
  assignments: list[None | (score, Candidate)] per region.
  discovery_notes: printable status strings.

  use_setlist: True = setlist + pool, gated to media >= 600s; False = setlist
  off (pool still runs on long media). Sub-10min media skips both: a single
  short song is never a concert setlist, picker auto-searches per region.
  setlist_titles: manual titles (--setlist-file), used instead of discovery.
  mc: detect spoken intermissions, keep out of queries/scoring, mark
  majority-spoken regions region.mc (never assigned a song)."""
  from .core.mc import classify_mc, mc_spans, mark_mc_regions
  from .core.regions import (detect_regions, print_region_table, region_query,
                             region_segments)
  from .core.providers import fetch_from_providers, media_tags
  from .core.align import score_candidates
  from .gui.picker import build_alternates

  notes = []

  regions = detect_regions(segments, audio, gap_threshold=25.0, min_region=60.0,
                           gate=0.03, dip=5.0, min_run=2.0)
  notes.append(f"detected {len(regions)} song regions")
  print(f"  detected {len(regions)} song regions")

  if not regions:
    return None

  mc_flags = classify_mc(segments) if mc else [False] * len(segments)
  mc_exclude = mc_spans(segments, mc_flags) if mc else []
  if mc_exclude:
    msg = f"mc: {sum(mc_flags)} spoken segments in {len(mc_exclude)} span(s)"
    print(f"  {msg}")
    notes.append(msg)
  print_region_table(regions)

  # total media duration: prefer audio samples, else last ASR segment end
  if audio is not None:
    duration_s = len(audio) / 16000
  elif segments:
    duration_s = max(e for _, e, _ in segments)
  else:
    duration_s = 0

  if duration_s < 600:
    msg = "short media (<10min): setlist/pool skipped"
    print(f"  {msg}")
    notes.append(msg)
    if mc:
      mark_mc_regions(regions, segments, mc_flags)
    return regions, [[] for _ in regions], [None] * len(regions), notes

  # artist hint: container tags first, else filename "Artist - Title"
  # (strip Live/Concert/Tour suffix); folder name unused
  tags = media_tags(path)
  alts = build_alternates(str(path), tags)
  artist_hint = ""
  if alts.get("artist"):
    raw = alts["artist"][0][0]
    artist_hint = re.sub(r'\s+(?:Live|Concert|Tour)\b.*', '', raw).strip()
    if artist_hint:
      print(f"  artist hint: {artist_hint}")

  # --- PRIMARY: setlist discovery + order-aware DP ---
  assignments = None
  region_all_scored = None
  if setlist_titles:
    tracklist = [(i + 1, t) for i, t in enumerate(setlist_titles)]
    msg = f"setlist: {len(tracklist)} tracks from file"
    print(f"  {msg}")
    notes.append(msg)
    assignments, region_all_scored = _setlist_dp_assign(
      regions, tracklist, segments, providers, artist_hint, fresh, notes,
      mc_exclude=mc_exclude, source_label="Setlist:Manual")
  elif use_setlist is not False:
    from .core.setlist import discover, hints_from_path
    hints = hints_from_path(path, artist_hint)
    tracklist, _, _ = discover(hints, fresh=fresh)
    if tracklist:
      assignments, region_all_scored = _setlist_dp_assign(
        regions, tracklist, segments, providers, artist_hint, fresh, notes,
        mc_exclude=mc_exclude)

  # --- FALLBACK: artist-pool greedy (only when no setlist found) ---
  if assignments is None:
    assignments = [None] * len(regions)
    region_all_scored = []
    artist_pool = []
    if artist_hint:
      print(f"  fallback: fetching artist discography...")
      artist_pool = fetch_from_providers(artist_hint, providers, fresh, limit=30)
      print(f"  artist pool: {len(artist_pool)} candidates ({sum(1 for c in artist_pool if c.lrc)} with LRC)")

    for i, region in enumerate(regions):
      rsegs = region_segments(segments, region, exclude=mc_exclude)
      media_dur_ms = int(region.duration * 1000)
      scored = score_candidates(artist_pool, rsegs, media_dur_ms)

      # region-specific query when artist pool misses
      if not scored or scored[0][0] < POOL_SIM_THRESHOLD:
        query = region_query(segments, region, exclude=mc_exclude)
        if artist_hint:
          query = f"{query} {artist_hint}"
        region_candidates = fetch_from_providers(query, providers, fresh)
        if region_candidates:
          region_scored = score_candidates(region_candidates, rsegs, media_dur_ms)
          all_scored = sorted(scored + region_scored, key=lambda x: -x[0])
          scored = _dedupe_by_title(all_scored)
      region_all_scored.append(scored)

    # greedy assignment: best score first, each title used once
    used_titles = set()
    triples = []
    for i, scored in enumerate(region_all_scored):
      for s, c in scored:
        if s >= POOL_SIM_THRESHOLD:
          triples.append((s, i, c))
    triples.sort(key=lambda x: -x[0])
    for s, i, c in triples:
      key = (c.title, c.artist)
      if assignments[i] is not None or key in used_titles:
        continue
      assignments[i] = (s, c)
      used_titles.add(key)
    msg = f"pool fallback: {sum(1 for a in assignments if a is not None)}/{len(regions)} assigned (threshold {POOL_SIM_THRESHOLD})"
    print(f"  {msg}")
    notes.append(msg)

  if mc:
    n_mc = mark_mc_regions(regions, segments, mc_flags, region_all_scored)
    if n_mc:
      # spoken region keeps no song: drop DP/greedy assignment
      for i, region in enumerate(regions):
        if region.mc:
          assignments[i] = None
      msg = f"mc: {n_mc}/{len(regions)} regions are spoken, not songs"
      print(f"  {msg}")
      notes.append(msg)

  return regions, region_all_scored, assignments, notes


def _align_one_region(region, region_idx, rsegs, audio, assignment, opts,
                      regions=None, segments=None, assigned=None):
  """Clean/align a single region. Returns (cues, meta) in absolute time.
  cues fall back to ASR (rsegs) on any skip path. Pure: no export/session.
  regions/segments/assigned: when given, align over an expanded window (see
  regions.expand_window) instead of own bounds, so a song split by a long
  instrumental or starting before a quiet intro still places whole.
  meta gains song_span, the placed extent deciding ownership.
  opts: place.AlignOpts."""
  from lyrickit import classify_lines, filter_credit_lines
  from .core.align import Cue
  from .core.place import run_chain

  if assignment is None:
    return list(rsegs), {"region_idx": region_idx, "strategy": "unassigned"}

  best_score, best = assignment
  user_picked = getattr(best, "user_picked", False)
  tag = " (user pick)" if user_picked else ""
  print(f"  matched: {best.title} - {best.artist} [{best.source}] sim={best_score:.2f}{tag}")
  region.candidate_idx = 0

  lines = best.lines
  if not lines:
    return list(rsegs), {"region_idx": region_idx, "strategy": "parse_empty"}
  # pin locale per region (one song per region), see _align_and_export
  from .core.romanize import detect, set_default_locale
  set_default_locale(detect("\n".join(t for _, t in lines)))

  verdicts = classify_lines(lines)
  clean_lines = filter_credit_lines(lines, verdicts)
  if not clean_lines:
    return list(rsegs), {"region_idx": region_idx, "strategy": "all_credits"}

  timed = [t for t, _ in clean_lines if t is not None]
  lrc_span = (max(timed) - min(timed)) if len(timed) > 1 else region.duration
  if regions is not None:
    from .core.regions import expand_window
    w0, w1 = expand_window(regions, region_idx, lrc_span, assigned=assigned)
  else:
    w0, w1 = region.start, region.end
  win_segs = ([(s, e, t) for s, e, t in segments if s >= w0 - 0.5 and e <= w1 + 0.5]
              if segments is not None else rsegs)

  region_audio = None
  if audio is not None:
    sr = 16000
    region_audio = audio[int(w0 * sr):int(w1 * sr)]

  segs_local = [(s - w0, e - w0, t) for s, e, t in win_segs]
  aligned_local, song_span_local, meta = run_chain(
    clean_lines, segs_local, region_audio, opts)

  # low-confidence guard: coarse_envelope with low sim -> keep ASR.
  # Skipped for user picks: human vouched for the match, so place LRC even
  # when ASR-vs-lyric similarity is low (ASR mishears singing).
  if (meta.get("strategy") == "coarse_envelope" and best_score < POOL_SIM_THRESHOLD
      and not user_picked):
    print(f"  low-confidence ({meta['strategy']}, sim={best_score:.2f}), keeping ASR")
    meta.update({"region_idx": region_idx, "strategy": "low_confidence_asr",
                 "title": best.title, "sim": round(best_score, 3)})
    return list(rsegs), meta

  offset_cues = []
  for c in aligned_local:
    if isinstance(c, Cue):
      offset_cues.append(Cue(c.start + w0, c.end + w0, c.text, c.confidence))
    else:
      offset_cues.append((c[0] + w0, c[1] + w0, c[2]))
  # never emit fewer than the ASR: a placement that yields nothing keeps ASR
  if not offset_cues:
    print(f"  placement produced no cues, keeping ASR")
    return list(rsegs), {"region_idx": region_idx, "strategy": "kept_asr",
                         "title": best.title, "sim": round(best_score, 3)}
  meta["region_idx"] = region_idx
  meta["title"] = best.title
  meta["sim"] = round(best_score, 3)
  meta["window"] = [round(w0, 3), round(w1, 3)]
  if song_span_local is not None:
    meta["song_span"] = [round(song_span_local[0] + w0, 3),
                         round(song_span_local[1] + w0, 3)]
  print(f"  placement: {meta.get('strategy', 'none')}")
  return offset_cues, meta


def _finalize_single_region(segments, audio, region, region_idx, assignment,
                            opts, regions=None):
  """Align one region for the GUI 'Align region' action. Returns (cues, meta)
  in absolute time, sorted. No export/session, caller merges into timeline.
  regions: full region list, enables window expansion around this region."""
  from .core.regions import region_segments, fmt_time
  rsegs = region_segments(segments, region)
  print(f"\n  --- region {region_idx+1}: {fmt_time(region.start)}-{fmt_time(region.end)} ---")
  cues, meta = _align_one_region(region, region_idx, rsegs, audio, assignment,
                                 opts, regions=regions, segments=segments)
  return sorted(cues, key=lambda c: c[0]), meta


# Share of an ASR segment that must sit inside a song span to count as already
# covered by lyrics. Containment is the wrong test: a segment straddling a span
# edge draws a duplicate line over the song's first/last cue. Text can't be
# trimmed by time, so a straddler is dropped or kept whole by larger share.
ASR_INSIDE_MAX = 0.5
SPAN_PAD = 0.5     # a fitted span's edge is only good to about this


def _inside_frac(seg, spans):
  """Fraction of an ASR segment's duration covered by the best-overlapping span.
  Spans padded by SPAN_PAD: edges are fitted not exact, so a fragment sitting
  right on one belongs to the song, not a separate transcript line."""
  dur = seg[1] - seg[0]
  if dur <= 0:
    return 1.0 if any(a - SPAN_PAD <= seg[0] <= b + SPAN_PAD for a, b in spans) else 0.0
  overlap = max((min(seg[1], b + SPAN_PAD) - max(seg[0], a - SPAN_PAD)
                 for a, b in spans), default=0.0)
  return max(overlap, 0.0) / dur


def _multi_song_finalize(path, segments, audio, regions, assignments,
                         romaji, opts, export=True, mc=True):
  """Per-region clean/align/merge from assignments, then optionally export+save.
  assignments: list[None | (score, Candidate)] per region.
  export=False (GUI 'Align all'): build cues + metas only, no SRT/session write.
  mc: spoken segments leave the lyric path, return as own 'mc' cues.
  Flags recomputed here (pure text work) so signature stays stable."""
  from .core.mc import classify_mc, mc_cues, mc_spans, strip_mc
  from .core.regions import region_segments, fmt_time
  from .core.export import export_srt
  from .core import session

  from .core.align import Cue

  mc_flags = classify_mc(segments) if mc else [False] * len(segments)
  mc_exclude = mc_spans(segments, mc_flags) if mc else []
  spoken = mc_cues(segments, mc_flags) if mc else []

  all_cues = []
  region_metas = []
  spans = []
  auto_picked = 0
  assigned = [a is not None for a in assignments]

  for i, region in enumerate(regions):
    rsegs = region_segments(segments, region, exclude=mc_exclude)
    print(f"\n  --- region {i+1}: {fmt_time(region.start)}-{fmt_time(region.end)} ---")
    if region.mc:
      print(f"  mc region (spoken), no lyrics")
      region_metas.append({"region_idx": i, "strategy": "mc"})
      continue
    if assignments[i] is None:
      print(f"  unassigned, keeping ASR")
    else:
      auto_picked += 1
    cues, meta = _align_one_region(region, i, rsegs, audio, assignments[i], opts,
                                   regions=regions, segments=segments,
                                   assigned=assigned)
    region_metas.append(meta)
    if meta.get("song_span"):
      spans.append(tuple(meta["song_span"]))
      all_cues.extend(c for c in cues if isinstance(c, Cue))
    # a region placing nothing contributes no cues; its ASR returns below
    # unless a neighbouring song's span already covers it

  # ASR survives only where lyrics don't already cover it: a region absorbed
  # into a neighbour's song (split song's tail, an MC break the take ran into)
  # must not leave raw transcript under the lyrics.
  # spoken segments return once as tagged mc cues, not also as plain ASR here
  outside = [seg for seg in strip_mc(segments, mc_exclude)
             if _inside_frac(seg, spans) < ASR_INSIDE_MAX]
  all_cues.extend(outside)
  spoken = [c for c in spoken if _inside_frac(c, spans) < ASR_INSIDE_MAX]
  all_cues.extend(spoken)
  all_cues.sort(key=lambda c: c[0])

  print(f"\n  auto-picked: {auto_picked}/{len(regions)} regions")
  print(f"  total cues: {len(all_cues)} ({len(outside)} ASR outside song spans,"
        f" {len(spoken)} mc)")

  if export:
    export_srt(path, all_cues, romaji=romaji)
    region_dicts = [r.to_dict() for r in regions]
    session.save(path, cues=all_cues, romaji=romaji,
                 regions=region_dicts, region_metas=region_metas,
                 mc_spans=mc_exclude,
                 placement={"multi_song": True, "regions": region_metas})
  return all_cues, region_metas


def _multi_song_headless(path, segments, audio, providers, fresh, romaji, opts,
                         use_setlist=True, mc=True, setlist_titles=None):
  """Headless multi-song: prepare → auto-assign → finalize."""
  from .core.export import export_srt
  from .core import session

  result = _multi_song_prepare(path, segments, audio, providers, fresh,
                               use_setlist=use_setlist, mc=mc,
                               setlist_titles=setlist_titles)
  if result is None:
    print(f"  no regions found, keeping pure ASR")
    export_srt(path, segments, romaji=romaji)
    session.save(path, cues=segments, romaji=romaji)
    return

  regions, per_region_scored, assignments, notes = result
  _multi_song_finalize(path, segments, audio, regions, assignments, romaji, opts,
                       mc=mc)


def _auto_suggest_regions(segments, audio):
  """Cheap region detection for auto-suggest banner. Returns region count."""
  from .core.regions import detect_regions
  regions = detect_regions(segments, audio, gap_threshold=25.0, min_region=60.0,
                           gate=0.03, dip=5.0, min_run=2.0)
  return len(regions)


def main():
  # pythonw: sys.stdout/stderr may be None
  if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  if sys.stderr is not None:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
  parser = argparse.ArgumentParser(description="Stamp official lyrics onto performance videos")
  parser.add_argument("targets", nargs="*", help="media files (with a cached ASR block); omit to open GUI empty")
  gui_default = _has_display()
  parser.add_argument("--gui", action=argparse.BooleanOptionalAction,
                      default=gui_default,
                      help="launch picker GUI (default when display available)")
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
                           "exported as their own grey cues (default ON)")
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
  args = parser.parse_args()

  export_mod.SRT_OFFSET_S = args.offset  # process-wide export defaults
  export_mod.SRT_LEAD_S = args.lead
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

  from .core.asr import load_asr
  from .core.providers import fetch_candidates, media_tags
  from .core.align import score_candidates
  from .core.export import export_srt
  from .core import session

  def _emit_pure_asr(path, segments, candidates, romaji, credit_toggles=None):
    """Export pure ASR, save session."""
    export_srt(path, segments, romaji=romaji)
    session.save(path, candidates=candidates, cues=segments, romaji=romaji,
                 credit_toggles=credit_toggles or {})
    print(f"  result: pure-ASR")

  from .core.romanize import set_default_locale

  for target in args.targets:
    path = Path(target)
    print(f"=== {path.name}")
    set_default_locale("")  # no locale leak from previous file

    # direct timeline open from existing session (--timeline, no --gui align)
    if args.timeline and not args.gui:
      from .gui.main_window import open_main_window
      open_main_window(session_only=str(path), providers=providers,
                       fresh=args.fresh, romaji=args.romaji, opts=opts)
      continue

    # re-export shortcut is headless-only: GUI always opens picker to refine
    # (fetch cache makes reopening free)
    sess = None if args.fresh else session.load(path)
    if sess and sess.get("cues") and not args.gui:
      written = session.reexport(path, sess)
      print(f"  result: re-export ({len(sess['cues'])} cues)")
      continue

    result = load_asr(path)
    if result is None:
      print(f"  skip (no ASR - transcribe first)")
      continue
    segments, _ = result

    # --- multi-song path ---
    if args.multi_song:
      audio, audio_path = _load_audio_once(path)
      if audio is not None and str(audio_path) != str(path):
        print(f"  audio: {Path(audio_path).name}")
      if args.gui:
        # prepare: detect + search + score (same as headless)
        result = _multi_song_prepare(path, segments, audio, providers,
                                     args.fresh, use_setlist=args.setlist,
                                     mc=args.mc, setlist_titles=setlist_titles)
        if result is None:
          print(f"  no regions found, keeping pure ASR")
          _emit_pure_asr(path, segments, [], args.romaji)
          continue
        regions, per_region_scored, assignments, notes = result
        tags = media_tags(path)
        media_dur_ms = tags.get("_duration_ms", 0)
        # MainWindow: picker + timeline + finalize all inside GUI
        from .gui.main_window import open_main_window
        exported = open_main_window(
          media_path=str(path), tags=tags, segments=segments,
          media_dur_ms=media_dur_ms, providers=providers, fresh=args.fresh,
          regions=regions, region_scored=per_region_scored,
          region_assignments=assignments, audio=audio,
          romaji=args.romaji, opts=opts)
        if exported:
          print(f"  result: multi-song (GUI)")
        else:
          print(f"  closed without export, keeping pure ASR")
          _emit_pure_asr(path, segments, [], args.romaji)
      else:
        _multi_song_headless(path, segments, audio, providers, args.fresh,
                             args.romaji, opts, use_setlist=args.setlist,
                             mc=args.mc, setlist_titles=setlist_titles)
      continue

    # fetch candidates (single ffprobe via media_tags)
    tags = media_tags(path)
    media_dur_ms = tags.get("_duration_ms", 0)
    candidates = fetch_candidates(path, providers, args.fresh, tags=tags)

    scored = score_candidates(candidates, segments, media_dur_ms)

    # --- GUI path: whole song = one full-span region, drive MainWindow ---
    if args.gui:
      audio, audio_path = _load_audio_once(path)
      if audio is not None and str(audio_path) != str(path):
        print(f"  audio: {Path(audio_path).name}")
      from .core.regions import Region
      if audio is not None:
        duration_s = len(audio) / 16000
      elif segments:
        duration_s = max(e for _, e, _ in segments)
      else:
        duration_s = media_dur_ms / 1000
      from .gui.main_window import open_main_window
      exported = open_main_window(
        media_path=str(path), tags=tags, segments=segments,
        media_dur_ms=media_dur_ms, providers=providers, fresh=args.fresh,
        regions=[Region(0.0, duration_s)], region_scored=[scored],
        region_assignments=[None], audio=audio,
        romaji=args.romaji, opts=opts)
      if exported:
        print(f"  result: single-song (GUI)")
      else:
        print(f"  closed without export, keeping pure ASR")
        _emit_pure_asr(path, segments, candidates, args.romaji)
      continue

    # auto-suggest: cheap region detection (headless hint only)
    if len(segments) > 50:
      audio_hint, _ = _load_audio_once(path)
      n_regions = _auto_suggest_regions(segments, audio_hint)
      if n_regions >= 2:
        print(f"  hint: {n_regions} song regions detected, consider --multi-song")

    # --- headless path ---
    if not scored:
      msg = "no lyrics found" if not candidates else "no lyrics with text found"
      print(f"  {msg}, keeping pure ASR")
      _emit_pure_asr(path, segments, candidates, args.romaji)
      continue

    best_score, best = scored[0]
    print(f"  top candidate: {best.title} [{best.source}] sim={best_score:.2f}")

    if best_score < SIMILARITY_THRESHOLD:
      print(f"  below threshold ({SIMILARITY_THRESHOLD}), keeping pure ASR")
      _emit_pure_asr(path, segments, candidates, args.romaji)
      continue

    status = _align_and_export(path, best, segments, candidates, args.romaji, opts)
    print(f"  result: {status}")


if __name__ == "__main__":
  main()
