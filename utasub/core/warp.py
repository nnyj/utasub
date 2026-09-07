"""Monotone piecewise-linear time warp: LRC time -> audio time.

Each segment is its own line (slope + offset) over line index. Two seam kinds:
  - gap insertion, discontinuous, for material the take adds or drops
    (extended instrumental, MC talk, dropped verse). Must land on real silence.
  - tempo change, continuous, for a take drifting against the studio cut.
    Needs no silence, only evidence the slope moved.

Slope locked to 1 on segments too short/sparse to measure, so noise can't
masquerade as tempo; a seam claims a tempo change only when both sides
measured one.
"""
from statistics import median

# --- knobs ---

# a jump must buy this many seconds of residual to be worth its own segment.
# Deliberately high: a fake gap wrecks a song, a missed 4s gap costs one line.
LAM = 25.0
MIN_JUMP = 4.0     # smaller offset changes are drift, not inserted material
NEG_MULT = 3.0     # cuts are rarer than inserts, charge more for them
MIN_SEG = 3        # anchors needed to define a segment
MIN_SEG_SPAN = 20.0  # LRC seconds a segment must cover: shorter is the fitter
                     # buying a jump to swallow bad anchors
ONSET_PRE = 1.5    # a line may start this early against a voiced run

# Slope band a segment's measured tempo must fall in: clipping a real slope
# makes the fitter buy fake gap insertions to fake the line it can't draw;
# too much freedom lets a sparse tail invent a tempo change.
# floor 0.90 covers slowest observed live take, ceiling 1.30 clears fastest
# (1.238) with margin
SLOPE_LO, SLOPE_HI = 0.90, 1.30
SLOPE_MIN_ANCHORS = 6
SLOPE_MIN_SPAN = 45.0
# Tempo change is continuous: the take drifts, doesn't skip. Needs no silence
# to hide in, only evidence the slope moved, so charged separately from a gap.
LAM_SLOPE = 3.0
SLOPE_BREAK_TOL = 0.75   # seconds of discontinuity still counted as continuous
SLOPE_BREAK_MIN = 0.03   # slope must move at least this much to be worth a break

# --- envelope helpers ---

def silence_gaps(runs, min_gap=2.0):
  """Unvoiced spans between voiced runs, long enough to hide inserted material."""
  return [(b, c) for (_, b), (c, _) in zip(runs, runs[1:]) if c - b >= min_gap]

def voiced_near(runs, t, pre=ONSET_PRE):
  """True when t sits inside a voiced run, or just before one starts."""
  return any(on - pre <= t <= off for on, off in runs)

# --- anchor filtering ---

def monotone_anchors(anchors):
  """Longest non-decreasing chain in audio time over ascending line index.
  Drops wrong-occurrence stamps on repeated lyrics and collapsed FA output."""
  n = len(anchors)
  if n < 2:
    return list(anchors)
  best = [1] * n
  prev = [-1] * n
  for i in range(n):
    for k in range(i):
      if anchors[k][1] <= anchors[i][1] and best[k] + 1 > best[i]:
        best[i], prev[i] = best[k] + 1, k
  e = max(range(n), key=lambda i: best[i])
  chain = []
  while e != -1:
    chain.append(anchors[e])
    e = prev[e]
  chain.reverse()
  return chain

CHAIN_TOL = 1.5      # per-anchor stamp jitter a pair's slope must tolerate
CHAIN_MIN_FRAC = 0.5  # below this the anchors are not one line and are kept as-is
CHAIN_SPLIT_FRAC = 0.8  # two chains covering this much of the set are a real seam

def linear_anchors(anchors, lrc_times, tol=CHAIN_TOL):
  """Longest chain whose every step implies a tempo a take could actually run.

  Stronger than monotone_anchors, which only requires audio time not go
  backwards: a wrong-occurrence stamp satisfies that easily, landing in order
  but seconds early, and a run of them drags the fitted slope. A stamp jumped
  to another repeat implies a tempo on both sides no singer could hold; this
  catches that.

  Band widened by tol so lines a few seconds apart are judged on the stamps'
  own accuracy, not a slope jitter dominates.

  Inserted material also breaks the chain, at the seam, and must not be
  thinned to one side of it. The cases part on where the break falls: inserted
  material splits the song at one point (discarded anchors form a contiguous
  block), wrong-occurrence stamps scatter through lines the good ones also
  cover. Leftovers are re-chained; only when the two occupy separate stretches
  are anchors handed back whole for the fit to seam.
  """
  n = len(anchors)
  if n < 3:
    return list(anchors)

  def longest(idx):
    """Indices of the longest plausible-tempo chain within idx."""
    best = [1] * len(idx)
    prev = [-1] * len(idx)
    for i in range(len(idx)):
      for k in range(i):
        dx = lrc_times[anchors[idx[i]][0]] - lrc_times[anchors[idx[k]][0]]
        dy = anchors[idx[i]][1] - anchors[idx[k]][1]
        if (dx > 0 and SLOPE_LO * dx - tol <= dy <= SLOPE_HI * dx + tol
            and best[k] + 1 > best[i]):
          best[i], prev[i] = best[k] + 1, k
    e = max(range(len(idx)), key=lambda i: best[i])
    out = []
    while e != -1:
      out.append(idx[e])
      e = prev[e]
    return out[::-1]

  keep = longest(list(range(n)))
  if len(keep) == n:
    return list(anchors)
  kept = set(keep)
  rest = [i for i in range(n) if i not in kept]
  if len(rest) >= MIN_SEG:
    other = longest(rest)
    if len(keep) + len(other) >= CHAIN_SPLIT_FRAC * n and len(other) >= MIN_SEG:
      a = [anchors[i][0] for i in keep]
      b = [anchors[i][0] for i in other]
      if max(a) < min(b) or max(b) < min(a):
        return list(anchors)
  if len(keep) < CHAIN_MIN_FRAC * n:
    return list(anchors)
  return [anchors[i] for i in keep]

def _theil_sen(xs, ys, min_dx=10.0):
  """(slope, offset) from pairwise slopes. Median tolerates up to half
  the anchors being wrong; least squares does not."""
  slopes = [(ys[b] - ys[a]) / (xs[b] - xs[a])
            for a in range(len(xs)) for b in range(a + 1, len(xs))
            if xs[b] - xs[a] > min_dx]
  if not slopes:
    return None
  sl = median(slopes)
  return sl, median(y - sl * x for x, y in zip(xs, ys))

def _fit_line(anchors, lrc_times, a, b):
  """(slope, offset, cost, measured) for anchors[a:b]. Slope is locked to 1
  unless the segment is long and well-populated enough to measure a tempo
  difference; measured says whether it was, which gates tempo-change points."""
  xs = [lrc_times[j] for j, _ in anchors[a:b]]
  ys = [t for _, t in anchors[a:b]]
  span = xs[-1] - xs[0]
  fit = None
  if len(xs) >= SLOPE_MIN_ANCHORS and span >= SLOPE_MIN_SPAN:
    fit = _theil_sen(xs, ys)
    if fit is not None and not (SLOPE_LO <= fit[0] <= SLOPE_HI):
      fit = None
  measured = fit is not None
  if fit is None:
    fit = (1.0, median(y - x for x, y in zip(xs, ys)))
  sl, off = fit
  return sl, off, sum(abs(y - (sl * x + off)) for x, y in zip(xs, ys)), measured

def _at(seg, t):
  """Audio time of an LRC time under one segment (slope, offset)."""
  return seg[0] * t + seg[1]

def _segment_fit(anchors, lrc_times, gaps):
  """Changepoint DP over anchor index. Returns [(a, b, slope, offset)]."""
  n = len(anchors)
  if n < MIN_SEG * 2:
    sl, off, _, _ = _fit_line(anchors, lrc_times, 0, n)
    return [(0, n, sl, off)]

  cost = {}
  for a in range(n):
    for b in range(a + MIN_SEG, n + 1):
      if lrc_times[anchors[b - 1][0]] - lrc_times[anchors[a][0]] < MIN_SEG_SPAN:
        continue
      sl, off, c, ms = _fit_line(anchors, lrc_times, a, b)
      cost[(a, b)] = (c, sl, off, ms)

  # D[(a, b)] = best total cost for anchors[:b] whose last segment is [a, b)
  D, back = {}, {}
  for b in range(MIN_SEG, n + 1):
    for a in range(0, b - MIN_SEG + 1):
      if (a, b) not in cost:
        continue
      c, sl, off, ms = cost[(a, b)]
      if a == 0:
        D[(a, b)], back[(a, b)] = c, None
        continue
      best, bk = None, None
      seam = lrc_times[anchors[a][0]]
      for a2 in range(0, a - MIN_SEG + 1):
        if (a2, a) not in D:
          continue
        _, sl2, off2, ms2 = cost[(a2, a)]
        # the jump is the discontinuity at the seam, not an offset difference:
        # with free slopes the offsets are not comparable on their own
        jump = _at((sl, off), seam) - _at((sl2, off2), seam)
        if abs(jump) <= SLOPE_BREAK_TOL:
          # continuous seam: only a genuine tempo change justifies the break;
          # a tempo neither side measured is not one
          if abs(sl - sl2) < SLOPE_BREAK_MIN or not (ms and ms2):
            continue
          pen = LAM_SLOPE
        else:
          if abs(jump) < MIN_JUMP:
            continue
          # a cut may not swallow more than the LRC interval it lands on
          if jump < 0 and -jump >= seam - lrc_times[anchors[a - 1][0]]:
            continue
          if not _jump_credible(gaps, _at((sl2, off2), lrc_times[anchors[a - 1][0]]),
                                _at((sl, off), seam), jump):
            continue
          pen = LAM * (NEG_MULT if jump < 0 else 1.0)
        tot = D[(a2, a)] + c + pen
        if best is None or tot < best:
          best, bk = tot, (a2, a)
      if best is not None:
        D[(a, b)], back[(a, b)] = best, bk

  ends = [(v, k) for k, v in D.items() if k[1] == n]
  if not ends:
    sl, off, _, _ = _fit_line(anchors, lrc_times, 0, n)
    return [(0, n, sl, off)]
  key = min(ends)[1]
  segs = []
  while key is not None:
    a, b = key
    _, sl, off, _ = cost[key]
    segs.append((a, b, sl, off))
    key = back[key]
  segs.reverse()
  return segs

def _jump_credible(gaps, t_prev, t_next, jump):
  """A jump is only real when unvoiced audio at the seam can account for it."""
  lo, hi = min(t_prev, t_next) - 5.0, max(t_prev, t_next) + 5.0
  return any(g1 > lo and g0 < hi and (g1 - g0) >= 0.6 * abs(jump)
             for g0, g1 in gaps)

def fit_warp(anchors, lrc_times, runs):
  """Fit the warp to [(line_idx, audio_time)] anchors.
  Returns (segments, diag) with segments = [(first_line_idx, slope, offset)]
  ascending, or None when there are too few anchors to fit anything.
  Each segment carries its own slope, so a take running a whole section slower
  than the studio cut is expressible without changepoints."""
  raw = len(anchors)
  anchors = linear_anchors(monotone_anchors(sorted(anchors)), lrc_times)
  if len(anchors) < 3:
    return None
  gaps = silence_gaps(runs)

  head = 0
  segs = _segment_fit(anchors, lrc_times, gaps)
  # head guard: a first segment placing line 0 in silence fitted on bad
  # anchors (wrong-occurrence matches in pre-song audio). Drop and refit.
  for _ in range(4 if runs else 0):
    if voiced_near(runs, _at(segs[0][2:], lrc_times[0])):
      break
    nxt = head + segs[0][1]
    if len(anchors) - nxt < MIN_SEG * 2:
      break
    head = nxt
    segs = _segment_fit(anchors[head:], lrc_times, gaps)
  kept = anchors[head:]
  # no snap-to-nearest-onset rescue here: the envelope can't tell singing from
  # crowd noise or an intro riff, so snapping a badly-fitted head onto the
  # first voiced run silently places a whole song one section early. A fit
  # this weak fails the residual gate in place._lrc_warp instead.
  out = [(kept[a][0], sl, off) for a, _, sl, off in segs]
  out[0] = (0, out[0][1], out[0][2])  # segment 0 owns every line before its first anchor
  resid = median([abs(kept[i][1] - _at((sl, off), lrc_times[kept[i][0]]))
                  for a, b, sl, off in segs for i in range(a, b)])
  jumps = []
  for k in range(1, len(out)):
    seam = lrc_times[out[k][0]]
    jumps.append(round(_at(out[k][1:], seam) - _at(out[k - 1][1:], seam), 2))
  # counted against the stamps handed in, so chain/monotone drops show too
  diag = {"anchors": len(kept), "dropped": raw - len(kept),
          "segments": len(out), "resid": round(resid, 3),
          "slopes": [round(sl, 4) for _, sl, _ in out],
          "head_ok": bool(runs) and voiced_near(runs, _at(out[0][1:], lrc_times[0])),
          "jumps": jumps}
  return out, diag

def _seg_at(segments, j):
  """Segment owning line index j."""
  k = 0
  while k + 1 < len(segments) and segments[k + 1][0] <= j:
    k += 1
  return segments[k]

def apply_warp(lrc_times, segments):
  """Warp LRC times to audio times. Monotone by construction: slopes stay
  positive and a cut can never exceed the LRC interval it lands on."""
  return [_at(_seg_at(segments, j)[1:], t) for j, t in enumerate(lrc_times)]

def warped_durations(lrc_times, segments):
  """Per-line sung duration under the warp: LRC interval scaled by that
  line's segment slope. Excludes any seam jump, since an inserted instrumental
  is silence, not a longer sung phrase. Last line holds 6s."""
  out = []
  for j in range(len(lrc_times)):
    sl = _seg_at(segments, j)[1]
    if j + 1 < len(lrc_times):
      out.append(sl * (lrc_times[j + 1] - lrc_times[j]))
    else:
      out.append(sl * 6.0)
  return out

def span_ok(placed, lrc_times, slack=60.0):
  """Reject a runaway fit that stretches the song far past its LRC length."""
  if len(placed) < 2:
    return True
  return placed[-1] - placed[0] <= (lrc_times[-1] - lrc_times[0]) * 1.5 + slack
