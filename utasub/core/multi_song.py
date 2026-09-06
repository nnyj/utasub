"""Multi-song flow: region detection, setlist/pool assignment, per-region
alignment, merge and export. Driven headless by cli.main and, with export off,
by the GUI workers."""
import re
from dataclasses import replace

# setlist-matched pairs accept lower sim (order constraint compensates)
SETLIST_SIM_THRESHOLD = 0.30
# artist-pool greedy fallback: high enough to reject noise
POOL_SIM_THRESHOLD = 0.55
# an auto-assignment this weak, or this close to the runner-up, is a coin flip
SUSPECT_SIM = 0.45
SUSPECT_MARGIN = 0.05

# region detection knobs, singing-tuned (looser than regions.detect_regions
# defaults, which serve short single-song media)
REGION_DETECT = dict(gap_threshold=25.0, min_region=60.0, gate=0.03, dip=5.0,
                     min_run=2.0)

# Share of an ASR segment that must sit inside a song span to count as already
# covered by lyrics. Containment is the wrong test: a segment straddling a span
# edge draws a duplicate line over the song's first/last cue. Text can't be
# trimmed by time, so a straddler is dropped or kept whole by larger share.
ASR_INSIDE_MAX = 0.5
SPAN_PAD = 0.5     # a fitted span's edge is only good to about this

# ━━━━━━ shared helpers ━━━━━━

def _note(notes, msg):
  """Print a status line and keep it for the GUI log."""
  print(f"  {msg}")
  if notes is not None:
    notes.append(msg)

def media_duration_s(audio, segments, media_dur_ms=0):
  """Media length in seconds: audio samples, else last ASR segment, else tags."""
  if audio is not None:
    return len(audio) / 16000
  if segments:
    return max(e for _, e, _ in segments)
  return media_dur_ms / 1000

def export_and_save(path, cues, romaji, edits=None, **save_kw):
  """Export the SRT set and write the session.
  edits: manual timings/deletions from a previous session, re-applied so a
  re-align never silently drops hand-corrected lines."""
  from . import export as export_mod, session
  if edits and any(edits.get(k) for k in session.EDIT_KINDS):
    cues = session.apply_manual_edits(cues, edits)
  # CLI --ass-* / --translation live on export module globals, pin them into the
  # session so the .ass written here and any later re-export agree
  save_kw.setdefault("ass_style", export_mod.ass_style())
  save_kw.setdefault("translation", export_mod.ASS_TRANSLATION)
  session.save(path, cues=cues, romaji=romaji, manual_edits=edits, **save_kw)
  session.reexport(path, session.load(path))
  return cues

def _load_audio_once(path):
  """Load audio from vocals stem or media, return (audio, audio_path)."""
  from .align import load_audio_16k, find_vocals
  audio_path = find_vocals(path) or path
  try:
    audio = load_audio_16k(audio_path)
    return audio, audio_path
  except Exception as e:
    print(f"  audio load failed ({e})")
    return None, audio_path

def detect_song_regions(segments, audio):
  """Song regions under the concert-tuned knobs."""
  from .regions import detect_regions
  return detect_regions(segments, audio, **REGION_DETECT)

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

# ━━━━━━ setlist DP ━━━━━━

def _setlist_dp_assign(regions, tracklist, segments, providers, artist_hint,
                       fresh, notes, mc_exclude=None,
                       source_label="Setlist:Wikipedia"):
  """Order-aware monotonic DP: match time-ordered regions to setlist tracks.
  Fetches lyrics per title, builds R×T score matrix, finds max-score monotonic
  matching allowing skips on both sides.
  Marks region.suspect (+ region.runner_up) on weak or near-tied wins.
  Returns (assignments, per_region_scored)."""
  from .regions import region_segments
  from .providers import fetch_from_providers
  from .align import score_candidates

  R = len(regions)
  T = len(tracklist)

  track_candidates = []  # list of list[Candidate]
  fetch_failures = 0
  for order, title in tracklist:
    query = f"{title} {artist_hint}" if artist_hint else title
    cands = fetch_from_providers(query, providers, fresh)
    if not any(c.lrc for c in cands):
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
      score, cand = score_mat[i - 1][j - 1]
      if not str(cand.source).startswith("Setlist:"):
        # relabel a copy: the same Candidate object also sits in other
        # regions' scored lists, and their provenance is not this one
        cand = replace(cand, source=source_label)
      assignments[i - 1] = (score, cand)
      _mark_suspect(regions[i - 1], score, cand, per_region_scored[i - 1])
      i -= 1
      j -= 1
    elif choice[i][j] == 1:
      j -= 1
    else:
      i -= 1

  matched = sum(1 for a in assignments if a is not None)
  n_suspect = sum(1 for r in regions if r.suspect)
  _note(notes, f"setlist DP: {matched}/{R} regions matched "
               f"({fetch_failures} tracks without lyrics)")
  if n_suspect:
    _note(notes, f"{n_suspect} region(s) uncertain (weak or near-tied match)")
  return assignments, per_region_scored

def _mark_suspect(region, score, cand, scored):
  """Flag an auto-pick a human ought to check: low score, or a runner-up on a
  different title within SUSPECT_MARGIN."""
  runner = next((c for s, c in scored
                 if (c.title, c.artist) != (cand.title, cand.artist)
                 and score - s <= SUSPECT_MARGIN), None)
  if score < SUSPECT_SIM or runner is not None:
    region.suspect = True
    region.runner_up = runner.title if runner is not None else ""

# ━━━━━━ prepare (detect + discover + assign) ━━━━━━

def _multi_song_prepare(path, segments, audio, providers, fresh,
                        use_setlist=True, mc=True, setlist_titles=None):
  """Detect regions, discover setlist (primary) or artist-pool greedy (fallback).
  Returns (regions, per_region_scored, assignments, discovery_notes, mc_flags) or None.
  assignments: list[None | (score, Candidate)] per region.
  discovery_notes: printable status strings.

  use_setlist: True = setlist + pool, gated to media >= 600s; False = setlist
  off (pool still runs on long media). Sub-10min media skips both: a single
  short song is never a concert setlist, picker auto-searches per region.
  setlist_titles: manual titles (--setlist-file), used instead of discovery.
  mc: detect spoken intermissions, keep out of queries/scoring, mark
  majority-spoken regions region.mc (never assigned a song)."""
  from .mc import classify_mc, mc_spans, mark_mc_regions
  from .regions import print_region_table, region_query, region_segments
  from .providers import build_alternates, fetch_from_providers, media_tags
  from .align import score_candidates

  notes = []
  regions = detect_song_regions(segments, audio)
  _note(notes, f"detected {len(regions)} song regions")
  if not regions:
    return None

  mc_flags = classify_mc(segments) if mc else [False] * len(segments)
  mc_exclude = mc_spans(segments, mc_flags) if mc else []
  if mc_exclude:
    _note(notes, f"mc: {sum(mc_flags)} spoken segments in {len(mc_exclude)} span(s)")
  print_region_table(regions)

  if media_duration_s(audio, segments) < 600:
    _note(notes, "short media (<10min): setlist/pool skipped")
    if mc:
      mark_mc_regions(regions, segments, mc_flags)
    return regions, [[] for _ in regions], [None] * len(regions), notes, mc_flags

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
    _note(notes, f"setlist: {len(tracklist)} tracks from file")
    assignments, region_all_scored = _setlist_dp_assign(
      regions, tracklist, segments, providers, artist_hint, fresh, notes,
      mc_exclude=mc_exclude, source_label="Setlist:Manual")
  elif use_setlist is not False:
    from .setlist import discover, hints_from_path
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
      _mark_suspect(regions[i], s, c, region_all_scored[i])
    n = sum(1 for a in assignments if a is not None)
    _note(notes, f"pool fallback: {n}/{len(regions)} assigned "
                 f"(threshold {POOL_SIM_THRESHOLD})")

  if mc:
    n_mc = mark_mc_regions(regions, segments, mc_flags, region_all_scored)
    if n_mc:
      # spoken region keeps no song: drop DP/greedy assignment
      for i, region in enumerate(regions):
        if region.mc:
          assignments[i] = None
      _note(notes, f"mc: {n_mc}/{len(regions)} regions are spoken, not songs")

  return regions, region_all_scored, assignments, notes, mc_flags

# ━━━━━━ per-region alignment ━━━━━━

def _candidate_index(scored, cand):
  """Position of the picked candidate in this region's scored list: what
  region.candidate_idx points back into. 0 when the list is not at hand."""
  for j, (_score, c) in enumerate(scored or []):
    if c is cand or (c.title, c.artist) == (cand.title, cand.artist):
      return j
  return 0

def _align_one_region(region, region_idx, rsegs, audio, assignment, opts,
                      regions=None, segments=None, assigned=None, scored=None):
  """Clean/align a single region. Returns (cues, meta) in absolute time.
  cues fall back to ASR (rsegs) on any skip path. Pure: no export/session.
  regions/segments/assigned: when given, align over an expanded window (see
  regions.expand_window) instead of own bounds, so a song split by a long
  instrumental or starting before a quiet intro still places whole.
  meta gains song_span, the placed extent deciding ownership.
  scored: this region's [(score, Candidate)] list, so the stored candidate_idx
  points at the candidate actually picked.
  opts: place.AlignOpts."""
  from lyrickit import classify_lines, filter_credit_lines
  from .align import Cue
  from .place import run_chain

  def skip(strategy, **extra):
    meta = {"region_idx": region_idx, "strategy": strategy}
    meta.update(extra)
    return list(rsegs), meta

  if assignment is None:
    return skip("unassigned")

  best_score, best = assignment
  user_picked = getattr(best, "user_picked", False)
  tag = " (user pick)" if user_picked else ""
  print(f"  matched: {best.title} - {best.artist} [{best.source}] sim={best_score:.2f}{tag}")
  region.candidate_idx = _candidate_index(scored, best)

  lines = best.lines
  if not lines:
    return skip("parse_empty")
  # pin locale per region (one song per region), see cli._align_and_export
  from .romanize import detect, set_default_locale
  set_default_locale(detect("\n".join(t for _, t in lines)))

  verdicts = classify_lines(lines)
  clean_lines = filter_credit_lines(lines, verdicts)
  if not clean_lines:
    return skip("all_credits")

  timed = [t for t, _ in clean_lines if t is not None]
  lrc_span = (max(timed) - min(timed)) if len(timed) > 1 else region.duration
  if regions is not None:
    from .regions import expand_window
    w0, w1 = expand_window(regions, region_idx, lrc_span, assigned=assigned)
  else:
    w0, w1 = region.start, region.end
  # overlap, not containment: a segment straddling the window edge is still
  # this window's audio and dropping it loses the line the singer started on
  win_segs = ([(s, e, t) for s, e, t in segments if e > w0 - 0.5 and s < w1 + 0.5]
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
      offset_cues.append(replace(c, start=c.start + w0, end=c.end + w0))
    else:
      offset_cues.append((c[0] + w0, c[1] + w0, c[2]))
  # never emit fewer than the ASR: a placement that yields nothing keeps ASR
  if not offset_cues:
    print(f"  placement produced no cues, keeping ASR")
    return skip("kept_asr", title=best.title, sim=round(best_score, 3))
  meta["region_idx"] = region_idx
  meta["title"] = best.title
  meta["sim"] = round(best_score, 3)
  meta["window"] = [round(w0, 3), round(w1, 3)]
  if region.suspect:
    meta["suspect"] = True
    meta["runner_up"] = region.runner_up
  if song_span_local is not None:
    meta["song_span"] = [round(song_span_local[0] + w0, 3),
                         round(song_span_local[1] + w0, 3)]
  print(f"  placement: {meta.get('strategy', 'none')}")
  return offset_cues, meta

def _finalize_single_region(segments, audio, region, region_idx, assignment,
                            opts, regions=None, scored=None):
  """Align one region for the GUI 'Align region' action. Returns (cues, meta)
  in absolute time, sorted. No export/session, caller merges into timeline.
  regions: full region list, enables window expansion around this region."""
  from .regions import region_segments, fmt_time
  rsegs = region_segments(segments, region)
  print(f"\n  --- region {region_idx+1}: {fmt_time(region.start)}-{fmt_time(region.end)} ---")
  cues, meta = _align_one_region(region, region_idx, rsegs, audio, assignment,
                                 opts, regions=regions, segments=segments,
                                 scored=scored)
  return sorted(cues, key=lambda c: c[0]), meta

# ━━━━━━ merge + export ━━━━━━

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
                         romaji, opts, export=True, mc=True, asr_fill=True,
                         edits=None, region_scored=None, mc_flags=None,
                         include_mc=True):
  """Per-region clean/align/merge from assignments, then optionally export+save.
  assignments: list[None | (score, Candidate)] per region.
  export=False (GUI 'Align all'): build cues + metas only, no SRT/session write.
  mc: spoken segments leave the lyric path, return as own 'mc' cues.
  asr_fill: keep transcript outside every song span as cues (junk filtered);
  off exports lyrics and MC only.
  region_scored: per-region [(score, Candidate)], so each region records the
  index of the candidate actually picked.
  mc_flags: spoken flags from _multi_song_prepare, classified here when absent.
  include_mc: off drops every MC cue; on also keeps in-song MC filling a gap."""
  from .mc import classify_mc, filter_junk, in_spans, mc_cues, mc_spans, strip_mc
  from .regions import region_segments, fmt_time
  from .align import Cue

  mc_flags = mc_flags or (classify_mc(segments) if mc else [False] * len(segments))
  mc_exclude = mc_spans(segments, mc_flags) if mc else []
  spoken = mc_cues(segments, mc_flags) if mc else []

  all_cues = []
  region_metas = []
  spans = []
  auto_picked = 0
  assigned = [a is not None for a in assignments]
  scored_lists = list(region_scored or [])

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
                                   assigned=assigned,
                                   scored=scored_lists[i] if i < len(scored_lists) else None)
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
  # between songs the stem is crowd noise, so the transcript there is invented
  n_raw = len(outside)
  if not asr_fill:
    # --no-asr-fill drops the transcript between songs, not the songs whose
    # lyrics never placed: a region with no span has nothing else to show
    unplaced = [(r.start, r.end) for r, m in zip(regions, region_metas)
                if not r.mc and not m.get("song_span")]
    outside = [seg for seg in outside
               if _inside_frac(seg, unplaced) >= ASR_INSIDE_MAX]
  outside = filter_junk(outside)
  all_cues.extend(outside)
  # MC between songs always shows; MC inside a song span shows only when
  # include_mc is on and it sits in a >=4s hole no placed cue covers (a live
  # break, not a mishear over a sung line)
  placed = [(c[0], c[1]) for c in all_cues]
  def _in_gap(c):
    if in_spans(c[0], c[1], placed):
      return False
    prev = max((b for a, b in placed if b <= c[0]), default=c[0])
    nxt = min((a for a, b in placed if a >= c[1]), default=c[1])
    return nxt - prev >= 4.0
  if include_mc:
    spoken = [c for c in spoken
              if _inside_frac(c, spans) < ASR_INSIDE_MAX or _in_gap(c)]
  else:
    spoken = []
  all_cues.extend(spoken)
  all_cues.sort(key=lambda c: c[0])

  print(f"\n  auto-picked: {auto_picked}/{len(regions)} regions")
  print(f"  total cues: {len(all_cues)} ({len(outside)} ASR outside song spans,"
        f" {n_raw - len(outside)} junk dropped, {len(spoken)} mc)")

  if export:
    all_cues = export_and_save(
      path, all_cues, romaji, edits=edits,
      regions=[r.to_dict() for r in regions], region_metas=region_metas,
      mc_spans=mc_exclude,
      placement={"multi_song": True, "regions": region_metas})
  return all_cues, region_metas

def _multi_song_headless(path, segments, audio, providers, fresh, romaji, opts,
                         use_setlist=True, mc=True, setlist_titles=None,
                         asr_fill=True, edits=None):
  """Headless multi-song: prepare → auto-assign → finalize.
  Returns (regions, region_metas), both empty when no region found."""
  result = _multi_song_prepare(path, segments, audio, providers, fresh,
                               use_setlist=use_setlist, mc=mc,
                               setlist_titles=setlist_titles)
  if result is None:
    print(f"  no regions found, keeping pure ASR")
    export_and_save(path, segments, romaji, edits=edits)
    return [], []

  regions, per_region_scored, assignments, notes, mc_flags = result
  _, region_metas = _multi_song_finalize(
    path, segments, audio, regions, assignments, romaji, opts, mc=mc,
    mc_flags=mc_flags, asr_fill=asr_fill, edits=edits, region_scored=per_region_scored)
  return regions, region_metas
