"""Placement-strategy chain: lrc_warp -> coarse_fa (envelope-only inside).
Each strategy: (lines, segments, audio, ...) -> (cues, song_span, meta) or None.
First success wins. meta records strategy name + diagnostics."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AlignOpts:
  """Session-global alignment knobs, threaded through the strategy chain as one
  object."""
  use_fa: bool = True


# --- knobs ---

MAX_RESID = 4.0      # median anchor residual above this rejects the fit outright
MIN_ANCHORS = 8      # fewer stamps than this and the warp is extrapolating
SNAP_WIN = 0.5       # how far a single line may be pulled onto a vocal onset
SNAP_TAU = 0.10      # distance at which an onset's strength is halved in scoring
SNAP_FLOOR = 0.02    # onset strength floor, relative to peak RMS
SILENCE_REACH = 2.0  # how far a line stranded in silence may be carried forward
# How far a gated CTC stamp may sit from the placement warp+snap already
# agreed on. Warp is fitted from these same stamps, so a stamp this far
# disagrees with every other stamp in the song. Tuned: tighter drops real
# bad-start corrections, looser drags in the outlier tail.
ADOPT_MAX = 2.0


# --- helpers ---

def _lrc_has_timestamps(lines):
  """True when LRC has line timestamps (not all None)."""
  return any(t is not None for t, _ in lines)


def _onset_strength(audio, sr=16000, frame_s=0.01, win=1024):
  """Per-frame spectral flux: summed positive change in magnitude spectrum.

  Fires on timbre change, not just loudness, so it marks a soft vocal entry
  over sustained backing where an RMS-rise curve would not.
  """
  import numpy as np

  hop = max(1, int(sr * frame_s))
  n = (len(audio) - win) // hop
  if n <= 1:
    return np.zeros(1), 1.0
  idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
  mag = np.abs(np.fft.rfft(audio[idx] * np.hanning(win), axis=1))
  flux = np.maximum(np.diff(mag, axis=0), 0).sum(axis=1)
  peak = float(flux.max())
  return flux, (peak if peak > 0 else 1.0)


def _rescue_from_silence(s, runs, reach=SILENCE_REACH):
  """Pull a start that landed in an unvoiced gap forward to the next voiced run.

  Per-line, deliberately weak: fires only on a line already in silence, only
  forward, only as far as reach, so it can never drag a line back over its
  predecessor or past its own phrase. A line already inside or just before
  voice is left alone.
  """
  from .warp import voiced_near
  if not runs or voiced_near(runs, s):
    return s
  nxt = next((on for on, _ in runs if s < on <= s + reach), None)
  return nxt if nxt is not None else s


def _adopt_fa(starts, fa_stamps, bound=ADOPT_MAX):
  """Take the global CTC stamp wherever the confidence gate kept one.

  Two cheap filters. Confidence gate (ctc_align.gate, applied by caller) drops
  lines whose own token log-probs say the aligner was guessing. The bound
  drops the rest of the tail: one monotonic path can't match the wrong repeat
  of a chorus, but can still smear a line across an instrumental, landing
  seconds from the warp every other stamp in the song agrees on. Everything
  the pair keeps is believed outright: the stamp measures this line's own
  attack, the snap only guesses at it.
  """
  if not fa_stamps:
    return list(starts)
  out = list(starts)
  for j, t in fa_stamps.items():
    if 0 <= j < len(out) and abs(t - out[j]) <= bound:
      out[j] = t
  for j in range(1, len(out)):
    out[j] = max(out[j], out[j - 1] + 0.05)
  return out


def _snap_lines(starts, audio, runs=None, win=SNAP_WIN, tau=SNAP_TAU,
                floor_rel=SNAP_FLOOR):
  """Pull each placed start onto the nearest strong vocal onset, independently
  per line, bounded to +/-win.

  Warp is smooth, so it can't express the per-line deviation of a live
  delivery: a singer enters a beat late here, early there. Each line clamped
  to its own small window, so the envelope can never move a whole song; worst
  case is one line off by win.

  Candidates scored by onset strength discounted by distance, so a line
  already on an attack stays put instead of being pulled to a louder
  neighbouring syllable. Symmetric window: an early start moves later as
  readily as a late one moves earlier.
  """
  frame_s = 0.01
  onset, peak = _onset_strength(audio, frame_s=frame_s)
  if len(onset) < 2:
    return list(starts)
  floor = peak * floor_rel

  out = []
  for s in starts:
    f0 = max(0, int((s - win) / frame_s))
    f1 = min(len(onset), int((s + win) / frame_s) + 1)
    best, best_score = s, 0.0
    for i in range(f0, f1):
      v = float(onset[i])
      if v < floor:
        continue
      t = i * frame_s
      score = v / (1.0 + abs(t - s) / tau)
      if score > best_score:
        best, best_score = t, score
    out.append(_rescue_from_silence(best, runs))
  for j in range(1, len(out)):
    out[j] = max(out[j], out[j - 1] + 0.05)
  return out


# --- strategies ---

def _lrc_warp(lines, segments, audio, use_fa=True, runs_lo=None, **_):
  """Place the LRC by a monotone warp fitted to global CTC anchors.
  One segment: steady offset. Extra segments: live take inserted/cut material
  and the fit found silence to hide the seam in. LRC intervals preserved
  inside every segment.
  Returns (cues, song_span, meta) or None."""
  if not _lrc_has_timestamps(lines) or audio is None:
    return None
  if not use_fa:
    return None

  texts = [t for _, t in lines]
  lrc_times = [t for t, _ in lines]
  n = len(texts)
  runs_lo = runs_lo or []

  from .anchors import collect
  from .ctc_align import ctc_available, gate
  from .warp import apply_warp, fit_warp, span_ok, warped_durations

  if not ctc_available():
    print("  lrc-warp: no CTC aligner (torchaudio missing)")
    return None
  need = MIN_ANCHORS

  # one global monotonic CTC pass: anchors come back in order and on their
  # own occurrence, so no candidate arbitration needed
  try:
    candidates, (ctc_starts, ctc_ends, scores) = collect(lrc_times, texts, audio)
  except Exception as e:
    print(f"  lrc-warp: ctc failed ({e})")
    return None
  if not candidates:
    print("  lrc-warp: no CTC anchors")
    return None

  fitted = None
  anchors, adiag = candidates[0]
  if len(anchors) >= need:
    res = fit_warp(anchors, lrc_times, runs_lo)
    # count anchors surviving monotone filtering + head guard, not raw stamps:
    # a free slope fitted on a handful of survivors swings the whole song
    if res is not None and res[1]["anchors"] >= need:
      res[1].update(adiag)
      if res[1]["resid"] > MAX_RESID:
        # anchors this scattered are not a measured take
        print(f"  lrc-warp: resid {res[1]['resid']}s over {MAX_RESID}s, rejecting")
      else:
        fitted = res
  if fitted is None:
    print("  lrc-warp: too few anchors to fit")
    return None
  warp_segs, diag = fitted

  # per-line stamps come from the same pass, gated on their own token
  # log-probs; ends decide where a cue before an instrumental break stops
  fa_stamps = gate(ctc_starts, scores)
  ends_hint = {j: t for j, t in ctc_ends.items() if j in fa_stamps}
  diag["gated"] = len(fa_stamps)

  starts = apply_warp(lrc_times, warp_segs)
  if not span_ok(starts, lrc_times):
    print(f"  lrc-warp: span {starts[-1] - starts[0]:.0f}s runs away from the "
          f"LRC's {lrc_times[-1] - lrc_times[0]:.0f}s, rejecting")
    return None

  durations = warped_durations(lrc_times, warp_segs)

  # snap first, then let a gated stamp override: the stamp measures this
  # line's own attack, the snap only picks the loudest onset near the warp,
  # and a loud mid-phrase syllable outscores the quiet real entry. Snap owns
  # lines with no surviving stamp.
  starts = _snap_lines(starts, audio, runs_lo)
  starts = _adopt_fa(starts, fa_stamps)

  dur = len(audio) / 16000
  starts = [min(max(0.0, s), dur - 0.5) for s in starts]
  for j in range(1, len(starts)):  # clamping the tail can only break order
    starts[j] = max(starts[j], starts[j - 1] + 0.05)

  from .align import build_cues, split_outside
  cues = build_cues(starts, texts, runs_lo, ends_hint=ends_hint,
                    confidence=["lrc"] * n, durations=durations)

  song_start, song_end = cues[0][0], cues[-1][1]
  pre, post = split_outside(segments, song_start, song_end)
  print(f"  lrc-warp: {diag.get('anchors', 0)} anchors, {diag['segments']} "
        f"segment(s), resid {diag.get('resid', '-')}s"
        + (f", slopes {diag['slopes']}" if diag.get("slopes") else "")
        + (f", jumps {diag['jumps']}" if diag.get("jumps") else ""))
  meta = {"strategy": "lrc_warp", **diag}
  return pre + cues + post, (round(song_start, 3), round(song_end, 3)), meta


def _coarse_fa(lines, segments, audio, use_fa=True, runs_lo=None, voiced_hi=None,
               **_):
  """Coarse char-level map, stamped by FA when it is available and useful.

  Fitness bar: FA must have stamped at least half the lines. Below that its
  stamps are noise on this song and the envelope alone places the text.
  **_ absorbs the uniform strategy signature."""
  from .align import align_lyrics
  aligned, song_span = align_lyrics(segments, lines, audio, use_fa=use_fa,
                                    runs_lo=runs_lo, voiced_hi=voiced_hi)
  if song_span is None:
    return None

  from .align import Cue
  fa_count = sum(1 for c in aligned if isinstance(c, Cue) and c.confidence == "fa")
  total = sum(1 for c in aligned if isinstance(c, Cue)
              and c.confidence in ("fa", "coarse", "interpolated"))

  if use_fa and audio is not None and total > 0 and fa_count < total * 0.5:
    print(f"  coarse-fa: only {fa_count}/{total} fa-stamped (<50%), envelope only")
    return _coarse_fa(lines, segments, audio, use_fa=False, runs_lo=runs_lo,
                      voiced_hi=voiced_hi)

  strategy = "coarse_fa" if (use_fa and audio is not None) else "coarse_envelope"
  meta = {
    "strategy": strategy,
    "fa_stamped": fa_count,
    "total_lines": total,
  }
  return aligned, song_span, meta


# --- chain ---

CHAIN = [
  ("lrc_warp", _lrc_warp),
  ("coarse_fa", _coarse_fa),
]


def run_chain(lines, segments, audio, opts=None, **_):
  """Run placement strategies in order, first non-None wins.
  Returns (cues, song_span, meta).
  opts: AlignOpts (use_fa), defaults to AlignOpts().
  Vocal envelopes computed once here, threaded into all strategies."""
  from .align import voiced_envelope, voiced_runs
  opts = opts or AlignOpts()

  runs_lo = []
  voiced_hi = None
  if audio is not None:
    env_lo = voiced_envelope(audio)
    if env_lo is not None:
      runs_lo = voiced_runs(env_lo)
    voiced_hi = voiced_envelope(audio, gate=0.08, local=True)

  for _name, fn in CHAIN:
    result = fn(lines, segments, audio, use_fa=opts.use_fa, runs_lo=runs_lo,
                voiced_hi=voiced_hi)
    if result is not None:
      return result
  # absolute fallback: return segments as-is
  return segments, None, {"strategy": "none"}
