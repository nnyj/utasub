"""Multi-song region detection from vocal envelope gaps, region-scoped ASR query extraction."""
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from .align import voiced_envelope, voiced_runs
from .romanize import is_cjk


@dataclass
class Region:
  """Time interval owning one candidate and its alignment."""
  start: float
  end: float
  candidate_idx: Optional[int] = None
  mc: bool = False   # spoken intermission: skip lyric fetch, setlist slot

  @property
  def duration(self):
    return self.end - self.start

  def to_dict(self):
    return {"start": round(self.start, 3), "end": round(self.end, 3),
            "candidate_idx": self.candidate_idx, "mc": self.mc}

  @staticmethod
  def from_dict(d):
    return Region(d["start"], d["end"], d.get("candidate_idx"),
                  bool(d.get("mc", False)))


# --- detection ---

def detect_regions(segments, audio, sr=16000,
                   gap_threshold=25.0, min_region=30.0,
                   gate=0.03, dip=3.0, min_run=2.0):
  """Detect song regions from vocal envelope gaps.
  Singing-tuned: moderate gate catches singing, dip merges instrumental breaks,
  gaps above threshold mark song boundaries.
  Returns list of Region ordered by start time."""
  if audio is None or len(audio) < sr * 10:
    return _regions_from_asr_gaps(segments, gap_threshold, min_region)
  voiced = voiced_envelope(audio, sr=sr, gate=gate)
  if voiced is None:
    return _regions_from_asr_gaps(segments, gap_threshold, min_region)
  runs = voiced_runs(voiced, dip=dip, min_run=min_run)
  if not runs:
    return []

  # gaps under threshold are the same song
  clusters = [(runs[0][0], runs[0][1])]
  for on, off in runs[1:]:
    prev_start, prev_end = clusters[-1]
    if on - prev_end < gap_threshold:
      clusters[-1] = (prev_start, off)
    else:
      clusters.append((on, off))

  # short clusters are noise (crowd claps), not songs
  regions = [Region(s, e) for s, e in clusters if e - s >= min_region]
  return regions


def _regions_from_asr_gaps(segments, gap_threshold, min_region):
  """Fallback: detect regions from ASR segment gaps when no audio."""
  if not segments:
    return []
  clusters = [(segments[0][0], segments[0][1])]
  for s, e, _ in segments[1:]:
    prev_start, prev_end = clusters[-1]
    if s - prev_end < gap_threshold:
      clusters[-1] = (prev_start, e)
    else:
      clusters.append((s, e))
  return [Region(s, e) for s, e in clusters if e - s >= min_region]


def expand_window(regions, idx, lrc_span, assigned=None, margin_min=30.0, pad=10.0):
  """Alignment window for a region: its own span plus margin, absorbing any
  contiguous unassigned neighbours whole, clamped to nearest assigned regions.
  Detection splits a song wherever a live take stretches an instrumental past
  the gap threshold, and misses quiet intros, so the region is a hint about
  where a song is, not a boundary. Ownership decided after alignment, from
  the placed span.
  assigned: per-region bool, defaults to candidate_idx being set.
  Returns (w0, w1) in absolute seconds."""
  r = regions[idx]
  if assigned is None:
    assigned = [q.candidate_idx is not None for q in regions]
  m = max(margin_min, lrc_span - r.duration)
  w0, w1 = r.start - m, r.end + m

  k = idx - 1
  while k >= 0 and not assigned[k]:
    w0 = min(w0, regions[k].start - pad)
    k -= 1
  if k >= 0:
    w0 = max(w0, regions[k].end)
  k = idx + 1
  while k < len(regions) and not assigned[k]:
    w1 = max(w1, regions[k].end + pad)
    k += 1
  if k < len(regions):
    w1 = min(w1, regions[k].start)
  return max(0.0, w0), max(w1, r.end)


# --- region-scoped query extraction ---

# Japanese particles/fillers to strip from n-gram queries
_JP_PARTICLES = set("は が の に を で と も か な よ ね へ や ら わ だ です ます"
                    " って けど から まで しか ばかり ながら たら なら ても でも"
                    " ああ うう ええ おお ふう".split())

_EN_STOPWORDS = set("the a an is are was were be been being have has had do does did"
                    " will would shall should may might can could i you he she it we they"
                    " me him her us them my your his its our their this that these those"
                    " and but or nor for yet so if then else when while as at by from in"
                    " into of on to with up out about over after before between through"
                    " during without along across against among around behind below beneath"
                    " beside besides beyond down inside near off onto opposite outside past"
                    " since toward under until upon not no oh yeah hey".split())


def _is_content_word(word):
  """True if word looks like a content word (not particle/filler)."""
  w = word.lower().strip()
  if not w or len(w) <= 1:
    return False
  if w in _JP_PARTICLES or w in _EN_STOPWORDS:
    return False
  return True


def _extract_ja_phrases(segments, region):
  """Extract short distinctive Japanese phrases from ASR segments in region.
  Prefers complete short segments (likely song phrases) over n-grams."""
  phrases = []
  for s, e, t in segments:
    if s < region.start - 1.0 or e > region.end + 1.0:
      continue
    clean = unicodedata.normalize("NFKC", t).strip()
    runs = re.findall(r'[぀-ゟ゠-ヿ一-鿿]{3,}', clean)
    for run in runs:
      if len(run) <= 12:
        phrases.append(run)
      else:
        phrases.append(run[:8])
  return phrases


def region_query(segments, region, max_phrases=2, exclude=None):
  """Extract search query from ASR segments in region.
  For Japanese: uses short distinctive phrases from ASR segments.
  For English: uses frequent content words.
  exclude: [(start, end), ...] spans to skip (MC speech pollutes the query).
  Returns query string, short enough for NetEase search."""
  if exclude:
    from .mc import strip_mc
    segments = strip_mc(segments, exclude)
  # collect ASR text within region bounds
  texts = [(s, e, t) for s, e, t in segments
           if s >= region.start - 1.0 and e <= region.end + 1.0]
  if not texts:
    return ""
  full = " ".join(t for _, _, t in texts)
  is_ja = is_cjk(full)

  if is_ja:
    phrases = _extract_ja_phrases(segments, region)
    if not phrases:
      # fallback: take a chunk from middle segment
      mid_seg = texts[len(texts) // 2][2]
      return mid_seg[:20].strip()

    freq = {}
    for p in phrases:
      freq[p] = freq.get(p, 0) + 1

    # rank by frequency, halved for phrases outside the distinctive 4-8 char band
    ranked = sorted(freq.items(),
                    key=lambda x: -x[1] * (1.0 if 4 <= len(x[0]) <= 8 else 0.5))
    query_phrases = [p for p, _ in ranked[:max_phrases]]
    return " ".join(query_phrases)
  else:
    tokens = [t for t in re.split(r'\W+', full) if t and _is_content_word(t)]
    if not tokens:
      mid = len(full) // 2
      return full[max(0, mid-15):mid+15].strip()
    freq = {}
    for t in tokens:
      freq[t.lower()] = freq.get(t.lower(), 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))
    return " ".join(w for w, _ in ranked[:3])


def region_segments(segments, region, exclude=None):
  """Return ASR segments within region bounds.
  exclude: [(start, end), ...] spans to drop (MC speech)."""
  out = [(s, e, t) for s, e, t in segments
         if s >= region.start - 0.5 and e <= region.end + 0.5]
  if exclude:
    from .mc import strip_mc
    out = strip_mc(out, exclude)
  return out


def fmt_time(seconds):
  """Format seconds as mm:ss."""
  m, s = divmod(int(seconds), 60)
  return f"{m:02d}:{s:02d}"


def print_region_table(regions):
  """Print region table to stdout."""
  print(f"  {'#':>3}  {'Span':>15}  {'Duration':>8}")
  print(f"  {'---':>3}  {'---------------':>15}  {'--------':>8}")
  for i, r in enumerate(regions):
    span = f"{fmt_time(r.start)}-{fmt_time(r.end)}"
    dur = f"{r.duration:.0f}s"
    tag = "  mc" if r.mc else ""
    print(f"  {i+1:>3}  {span:>15}  {dur:>8}{tag}")
