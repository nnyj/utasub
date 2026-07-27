"""Coarse char-level romaji-space map onto timed ASR, vocal envelope snap,
candidate scoring, forced alignment integration."""
import difflib
import re
import unicodedata
import subprocess
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .romanize import detect, romanize


SIMILARITY_THRESHOLD = 0.35


@dataclass
class Cue:
  """Timed lyric line. confidence: 'fa', 'coarse', 'interpolated', 'lrc', or None."""
  start: float
  end: float
  text: str
  confidence: Optional[str] = None

  def __iter__(self):
    """Iterate as (start, end, text[, confidence]) tuple."""
    yield self.start
    yield self.end
    yield self.text
    if self.confidence is not None:
      yield self.confidence

  def __getitem__(self, idx):
    return list(self)[idx]

  def __len__(self):
    return 4 if self.confidence is not None else 3

# --- romaji normalization ---

def romaji_key(texts, locale=None):
  """Normalize texts to lowercase ASCII romaji, stripping punctuation.
  locale None detects once over all texts, so scoring never reads the song-level
  default and kanji-only lines don't get per-line-detected as Chinese."""
  if locale is None:
    locale = detect(" ".join(texts))
  return [re.sub(r"[^0-9a-z]", "", romanize(t, locale=locale).lower()) for t in texts]


# --- candidate scoring ---

def score_candidate(candidate, segments, media_duration_ms=0, asr_key=None):
  """Score a candidate by text similarity (romaji-space SequenceMatcher)
  and duration proximity. Returns float 0-1.
  asr_key: precomputed romaji key for ASR text (avoids re-converting per candidate)."""
  if not candidate.lrc:
    return 0.0
  lyric_text = " ".join(t for _, t in candidate.lines)
  if not lyric_text:
    return 0.0

  if asr_key is None:
    asr_text = " ".join(t for _, _, t in segments)
    if not asr_text:
      return 0.0
    asr_key = romaji_key([asr_text])[0]

  lyr_key = romaji_key([lyric_text])[0]
  text_sim = difflib.SequenceMatcher(None, lyr_key, asr_key, autojunk=False).ratio()

  dur_sim = 1.0
  if candidate.duration_ms > 0 and media_duration_ms > 0:
    ratio = candidate.duration_ms / media_duration_ms
    dur_sim = max(0.0, 1.0 - abs(1.0 - ratio))

  return text_sim * 0.8 + dur_sim * 0.2


def score_candidates(candidates, segments, media_dur_ms):
  """Score all candidates with LRC text against ASR segments.
  Returns sorted [(score, candidate)] descending."""
  asr_text = " ".join(t for _, _, t in segments)
  asr_key = romaji_key([asr_text])[0] if asr_text else ""
  scored = []
  for c in candidates:
    if not c.lrc:
      continue
    s = score_candidate(c, segments, media_dur_ms, asr_key=asr_key)
    scored.append((s, c))
  scored.sort(key=lambda x: -x[0])
  return scored


# --- coarse alignment ---

def coarse_map(segments, texts):
  """Global char-level alignment of lyric text onto the ASR transcript in romaji space.
  Each ASR char gets a time interpolated inside its segment; SequenceMatcher blocks
  pin lyric chars to those times, unmatched stretches interpolate.
  Returns (starts, matched) per line, or None when nothing matches."""
  asr_chars, asr_times = _asr_char_stream(segments)

  keys = romaji_key(texts)
  line_pos = [0]
  for k in keys:
    line_pos.append(line_pos[-1] + len(k))
  lyr = "".join(keys)

  sm = difflib.SequenceMatcher(None, lyr, "".join(asr_chars), autojunk=False)
  blocks = [b for b in sm.get_matching_blocks() if b.size >= 4]
  if not blocks:
    return None

  pts = []
  for a, b, size in blocks:
    pts.append((a, asr_times[b]))
    pts.append((a + size - 1, asr_times[b + size - 1]))
  rate = (pts[-1][1] - pts[0][1]) / max(pts[-1][0] - pts[0][0], 1)

  def t_at(c):
    if c <= pts[0][0]:
      return max(0.0, pts[0][1] - (pts[0][0] - c) * rate)
    for (c0, t0), (c1, t1) in zip(pts, pts[1:]):
      if c <= c1:
        return t0 + (c - c0) * (t1 - t0) / max(c1 - c0, 1)
    return pts[-1][1] + (c - pts[-1][0]) * rate

  starts = [t_at(line_pos[j]) for j in range(len(texts))]
  matched = [any(a < line_pos[j + 1] and a + size > line_pos[j]
                 for a, _, size in blocks)
             for j in range(len(texts))]
  return starts, matched


# --- coarse helpers ---

def _safe_romaji_key(texts):
  """romaji_key per item, empty string on failure (e.g. Chinese chars cutlet chokes on).
  Locale detected once over the whole batch, not per item."""
  locale = detect(" ".join(texts))
  out = []
  for t in texts:
    try:
      out.append(romaji_key([t], locale=locale)[0])
    except Exception:
      out.append("")
  return out


def _asr_char_stream(segments):
  """Timed ASR romaji char stream: each segment's romaji key expanded to per-char
  times interpolated inside the segment span. Returns (chars, times)."""
  chars = []
  times = []
  seg_keys = _safe_romaji_key([t for _, _, t in segments])
  for (s, e, _), k in zip(segments, seg_keys):
    for i, ch in enumerate(k):
      chars.append(ch)
      times.append(s + (e - s) * i / max(len(k) - 1, 1))
  return chars, times


# --- vocal envelope ---

def frame_rms(audio, frame_s, sr=16000):
  """Per-frame RMS over non-overlapping frame_s-second frames.
  Returns array (empty when audio shorter than one frame)."""
  frame = int(sr * frame_s)
  n = len(audio) // frame
  if n == 0:
    return np.array([])
  return np.sqrt((audio[:n * frame].reshape(n, frame).astype(np.float64) ** 2).mean(axis=1))


def voiced_envelope(audio, sr=16000, gate=0.02, local=False):
  """RMS-gated voiced/silent per 50ms frame. Low gate catches tails,
  high gate marks singing onsets. local=True gates against rolling loudness."""
  rms = frame_rms(audio, 0.05, sr)
  n = len(rms)
  if n == 0:
    return None
  if not local:
    return rms > max(float(rms.max()) * gate, 1e-3)
  blocks = rms[:(n // 20) * 20].reshape(-1, 20).max(axis=1)
  k = 10
  ref = np.array([blocks[max(0, i - k):i + k + 1].max() for i in range(len(blocks))])
  ref = np.repeat(ref, 20)
  ref = np.concatenate([ref, np.full(n - len(ref), ref[-1] if len(ref) else 0)])
  ref = np.maximum(ref, float(rms.max()) * 0.1)
  return rms > np.maximum(ref * gate, 1e-3)


def voiced_runs(voiced, dip=0.45, min_run=0.2):
  """Voiced (onset, offset) times in seconds; gaps shorter than dip merged."""
  runs = []
  start = None
  last = None
  for i, v in enumerate(voiced):
    if not v:
      continue
    if start is None:
      start = i
    elif (i - last) * 0.05 >= dip:
      runs.append((start * 0.05, (last + 1) * 0.05))
      start = i
    last = i
  if start is not None:
    runs.append((start * 0.05, (last + 1) * 0.05))
  return [(s, e) for s, e in runs if e - s >= min_run]


# --- envelope placement (--no-fa path) ---

def walk_end(s, e0, runs):
  """Voice stop for a line starting at s: merge voiced runs across breath gaps < 1s."""
  cur_end = s
  for on, off in runs:
    if off <= s:
      continue
    if on >= e0 or on - cur_end >= 1.0:
      break
    cur_end = off
  return cur_end


def split_outside(segments, start, end):
  """ASR segments fully outside the song span: (pre, post).
  pre ends at/before start, post begins at/after end."""
  pre = [seg for seg in segments if seg[1] <= start]
  post = [seg for seg in segments if seg[0] >= end]
  return pre, post


CUE_GAP_S = 0.0  # gap left between a cue end and the next start (0 = seamless)
UNVOICED_HOLD = 4.0   # cue length when the envelope finds no voice at the start
LONG_GAP_S = 10.0     # an LRC interval this long is a break, not a sung phrase
LONG_GAP_HOLD = 8.0   # ceiling for a line that precedes such a break
# A line a break follows has no reliable length in the LRC (its interval covers
# the break), so its end rests on the envelope, which happily merges sustained
# backing or crowd energy. These bound it by what the singer could plausibly
# have sung, at the song's own measured character rate.
RATE_GUARD_RATIO = 2.5
RATE_GUARD_SLACK = 3.0
# The same guard in the other direction: below this share of the time a line's
# words need, the envelope has not found the end of the phrase, it has lost the
# voice inside it. Deliberately low, since the trim is right far more often than
# it is wrong and a loose floor would undo it.
RATE_FLOOR = 0.3


def build_cues(starts, texts, runs, ends_hint=None, confidence=None,
               durations=None):
  """Cues from monotonic starts.
  durations: expected sung length per line, the LRC interval scaled by the warp
  slope. When given it sets the duration and the envelope only trims it where
  the audio is clearly silent; FA/envelope ends can shorten a cue but never
  lengthen one. Without it (coarse path) the envelope walk decides.
  confidence: optional list parallel to starts, stored in 4th cue element."""
  out = []
  # chars/sec from lines whose LRC interval is their real sung length (no break
  # after). Sanity-checks break lines only.
  chars = [len(re.sub(r"[^0-9a-z぀-ヿ一-鿿]", "",
                      unicodedata.normalize("NFKC", t).lower())) for t in texts]
  rate = None
  if durations is not None:
    per = sorted(chars[j] / durations[j] for j in range(len(texts))
                 if durations[j] > 0.2 and durations[j] <= LONG_GAP_S and chars[j] > 2)
    if len(per) >= 5:
      rate = per[len(per) // 2]

  for j, s in enumerate(starts):
    e0 = starts[j + 1] - CUE_GAP_S if j + 1 < len(starts) else s + 6.0
    conf = confidence[j] if confidence is not None else None
    if durations is None:
      e = max((ends_hint or {}).get(j, 0.0), walk_end(s, e0, runs))
      e = min(e0, e) if e > s else min(e0, s + 8.0)
      out.append(Cue(s, max(e, s + 0.2), texts[j], conf))
      continue
    d = durations[j]
    e_lrc = s + d
    # walk a little past the LRC end so a held note still reads as voiced
    e_env = max(walk_end(s, e_lrc + 1.0, runs), (ends_hint or {}).get(j, 0.0))
    voiced = e_env > s + 0.4
    if not voiced:
      # no voice at start: hold short cue, not full LRC interval (an intro or
      # isolated line can run tens of seconds)
      e = min(e_lrc, s + UNVOICED_HOLD)
    elif e_lrc - e_env > 1.0:
      e = e_env  # envelope trims: voice stopped well before LRC says
      if rate and chars[j] > 2 and (e_env - s) < RATE_FLOOR * (chars[j] / rate):
        # unless too early to be real: a held/breathy note falls under the
        # gate mid-phrase, walk returns a fraction of the line. Floor on the
        # trim, not a replacement for it.
        e = e_lrc
    else:
      e = e_lrc
    # line before a long LRC gap (instrumental, break) must not stretch to
    # fill it just because the stem still reads as voiced
    if d > LONG_GAP_S:
      e = min(e, max(e_env, s + LONG_GAP_HOLD))
      if rate and chars[j] > 2:
        est = chars[j] / rate
        if (e - s) > max(RATE_GUARD_RATIO * est, est + RATE_GUARD_SLACK):
          e = s + est  # far longer than these words take to sing: envelope lied
    e = min(e, e0)
    out.append(Cue(s, max(e, min(s + min(d, 0.8), max(e0, s + 0.05))),
                   texts[j], conf))
  return out


def place_by_envelope(starts, texts, voiced, runs, ends_hint=None, confidence=None):
  """Snap starts to vocal onsets, build cues. The --no-fa placement path."""
  if voiced is None:
    return build_cues(starts, texts, runs or [], ends_hint=ends_hint, confidence=confidence)
  onsets = [s for s, _ in voiced_runs(voiced)]
  snapped = []
  for j, s in enumerate(starts):
    if j and s - starts[j - 1] < 1.0:
      snapped.append(max(snapped[-1] + 0.05, snapped[-1] + (s - starts[j - 1])))
      continue
    if onsets:
      near = min(onsets, key=lambda o: abs(o - s))
      if abs(near - s) <= 0.4:
        s = near
      elif not voiced[min(max(int(s / 0.05), 0), len(voiced) - 1)]:
        nxt = next((o for o in onsets if s < o <= s + 1.5), None)
        if nxt is not None:
          s = nxt
    snapped.append(max(snapped[-1] + 0.05, s) if snapped else max(0.0, s))
  return build_cues(snapped, texts, runs or [], ends_hint=ends_hint, confidence=confidence)


# --- audio loading ---

def load_audio_16k(path):
  """Decode via ffmpeg to mono 16k float32."""
  cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
  # CREATE_NO_WINDOW: suppress console flash under pythonw GUI
  flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
  raw = subprocess.run(cmd, capture_output=True, check=True,
                       creationflags=flags).stdout
  return np.frombuffer(raw, dtype=np.float32)


def stem_cache_dir():
  """Vocal-stem cache, out of the media folder to keep it tidy.
  %LOCALAPPDATA%/utasub/stems (temp dir fallback)."""
  import os
  import tempfile
  from pathlib import Path
  base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
  d = Path(base) / "utasub" / "stems"
  d.mkdir(parents=True, exist_ok=True)
  return d


def find_vocals(path):
  """Vocal stem file <name>.vocals.<ext>: beside the source first, then the
  stem cache. Cache keys on basename, so same-named files in different folders
  collide, acceptable for a single-user tool."""
  from pathlib import Path
  p = Path(path)
  want = f"{p.stem}.vocals".lower()
  local = next((f for f in p.parent.iterdir()
                if f.is_file() and f.stem.lower() == want), None)
  if local:
    return local
  cache = stem_cache_dir()
  return next((f for f in cache.iterdir()
               if f.is_file() and f.stem.lower() == want), None)


# --- top-level align ---

def align_lyrics(segments, lines, audio=None, use_fa=False,
                 runs_lo=None, voiced_hi=None):
  """Coarse align lyrics onto ASR + envelope snap + optional forced alignment.
  Returns (cues, song_span_or_None) where cues = [(start, end, text[, confidence]), ...].
  Confidence: 'fa' (fa-stamped), 'coarse' (coarse-only), 'interpolated' (non-decreasing fix).
  Song span replaces ASR text; ASR outside survives.
  runs_lo/voiced_hi: precomputed envelopes (avoids recomputation across strategies)."""
  texts = [t for _, t in lines]
  cm = coarse_map(segments, texts)
  if cm is None:
    print("  lyrics: no overlap with ASR transcript, keeping ASR output")
    return segments, None
  starts, matched = cm

  if runs_lo is None:
    runs_lo = []
  if voiced_hi is None and audio is not None:
    voiced_hi = voiced_envelope(audio, gate=0.08, local=True)
  if audio is not None:
    dur = len(audio) / 16000
    starts = [min(s, dur - 1.0) for s in starts]

  # trim unsung edge lines (credits/headers) when envelope available
  def sung(j):
    s = starts[j]
    e = starts[j + 1] if j + 1 < len(starts) else s + 4.0
    return any(on < max(e, s + 1.0) and off > s for on, off in runs_lo)

  lo, hi = 0, len(texts)
  while lo < hi and not matched[lo] and runs_lo and not sung(lo):
    lo += 1
  while hi > lo and not matched[hi - 1] and runs_lo and not sung(hi - 1):
    hi -= 1
  if lo or hi < len(texts):
    print(f"  lyrics: dropped {lo + len(texts) - hi} unsung edge line(s)")
  texts = texts[lo:hi]
  starts = starts[lo:hi]
  matched = matched[lo:hi]
  if not texts:
    return segments, None

  # forced alignment pass
  fa_starts, fa_ends = {}, {}
  confidence = ["coarse"] * len(texts)
  if use_fa and audio is not None:
    try:
      from .fa import force_align_lines
      fa_starts, fa_ends = force_align_lines(starts, texts, audio)
      fa_count = len(fa_starts)
      print(f"  fa: stamped {fa_count}/{len(texts)} lines")
      # replace coarse starts where fa passed plausibility gate
      for j in range(len(texts)):
        if j in fa_starts:
          starts[j] = fa_starts[j]
          confidence[j] = "fa"
    except Exception as e:
      print(f"  fa: failed ({e}), keeping coarse")

  # force non-decreasing after fa merge
  for j in range(1, len(starts)):
    if starts[j] < starts[j - 1]:
      starts[j] = starts[j - 1] + 0.05
      if confidence[j] != "coarse":
        confidence[j] = "interpolated"

  # envelope snap placement, fa ends as hints: the envelope walks the cue end,
  # the fa end wins when it lands later
  out = place_by_envelope(starts, texts, voiced_hi, runs_lo,
                          ends_hint=fa_ends, confidence=confidence)

  song_start = out[0][0]
  song_end = out[-1][1]
  pre, post = split_outside(segments, song_start, song_end)
  print(f"  lyrics: {sum(matched)}/{len(texts)} lines matched ASR; "
        f"{len(pre) + len(post)} ASR segments kept outside song span")
  return pre + out + post, (round(song_start, 3), round(song_end, 3))
