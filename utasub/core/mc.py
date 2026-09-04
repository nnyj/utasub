"""MC (spoken intermission) detection from the cached ASR transcript.

Text-only signals, no audio: ASR block already there, stage patter reads
differently from singing on the page. Keeps MC speech out of song matching,
emits it as its own subtitle track.
"""
import re
import unicodedata

# --- signal weights, summed per segment; >= MC_THRESHOLD marks the segment MC ---
W_PUNCT = 1.0    # Qwen punctuates spoken lines, sung lines mostly come through bare
W_POLITE = 1.2   # polite copula / greeting / stage-patter markers
W_FILLER = 0.7   # laughter and hesitation fillers, absent from lyrics
W_RATE = 0.6     # speech packs more characters per second than singing
MC_THRESHOLD = 1.4

PUNCT_CHARS = "。、！？，,.!?…‥・「」『』"
PUNCT_DENSITY_HI = 0.05   # punctuation share of characters that reads as speech
RATE_FAST = 7.0           # chars/sec above this is patter, not a sung phrase
MIN_CHARS = 5             # shorter text carries no usable signal
MIN_DUR = 0.2             # guard against zero-length ASR stamps in the rate signal

# polite/greeting/announcement markers: an MC block is talk *to* the room
_POLITE = ("です", "ます", "ました", "ません", "でしょう", "ありがとう",
           "ございます", "みなさん", "皆さん", "こんばんは", "こんにちは",
           "次の曲", "最後の曲", "よろしく", "お願い", "本当に", "今日は")
# fillers and transcribed laughter
_FILLER = ("えー", "あの", "あのー", "なんか", "ええと", "うーん", "笑",
           "ハハ", "はは", "あはは", "うん", "はい")

# --- junk-cue filter: ASR invents text over crowd noise, cheers and chants ---
JUNK_MIN_CHARS = 4        # under 4 chars is weak evidence on its own, see is_junk_cue
JUNK_MIN_DUR = 0.35       # a stamp this short is a degenerate ASR fragment
JUNK_CPS_HI = 22.0        # chars/sec above this is transcription garbage over noise
JUNK_CPS_LO = 0.5         # long span with almost no text is crowd noise, not a line
JUNK_REPEAT_MIN = 3       # a unit must recur 3x+ to read as a chant, not a word
JUNK_REPEAT_COVER = 0.9   # the repeated unit must fill ~all of the text
JUNK_MIN_SIGNALS = 2      # weak signals must agree, one alone kills real lyrics

# MC segments this far apart still belong to the same spoken block
SPAN_GAP = 8.0
# share of a region's segments that must be MC for the region to count as MC
MC_MAJORITY = 0.6

def _norm(text):
  return unicodedata.normalize("NFKC", (text or "").strip())

def _hits(text, markers):
  return sum(1 for m in markers if m in text)

def _repeated_unit(text):
  """True when text is one short unit (1-3 chars) tiled across nearly all of it,
  the chant/cheer shape like ワーワーワーワー or ah ah ah ah."""
  t = text.replace(" ", "")
  if not t:
    return False
  if len(set(t)) == 1 and len(t) >= JUNK_REPEAT_MIN:
    return True
  for unit in range(1, 4):
    if len(t) < unit * JUNK_REPEAT_MIN:
      continue
    head = t[:unit]
    reps = len(t) // unit
    if head * reps == t[:unit * reps] and (unit * reps) / len(t) >= JUNK_REPEAT_COVER:
      return True
  return False

def _strip_punct(text):
  """Text without leading/trailing punctuation, so `アンコール!` and `アンコール`
  read as the same line."""
  return text.strip(PUNCT_CHARS + " ")

def _near_repeat(text, prev):
  """Repeat of the previous kept cue: equal once end punctuation is off, or a
  prefix overlap when both texts are chant-shaped. A real line that merely
  starts like the one before it (ありがとう after ありがとうございました) is not
  a repeat."""
  if not prev:
    return False
  a, b = _strip_punct(text), _strip_punct(prev)
  if a == b:
    return True
  if not (_repeated_unit(a) and _repeated_unit(b)):
    return False
  return a.startswith(b) or b.startswith(a)

def junk_signals(seg):
  """Weak marks that an ASR segment is invented, not sung. Each is common enough
  in real singing to be wrong on its own, so is_junk_cue wants JUNK_MIN_SIGNALS of them."""
  start, end, text = seg[0], seg[1], _norm(seg[2])
  dur = end - start
  hits = []
  if len(text) < JUNK_MIN_CHARS:
    hits.append("short")
  if len(text) / max(dur, MIN_DUR) <= JUNK_CPS_LO:
    hits.append("slow")
  if dur < JUNK_MIN_DUR:
    hits.append("tiny")
  return hits

def is_junk_cue(seg, prev_text=None):
  """True when an ASR segment is junk invented over crowd noise, cheers or
  chants, and is dropped before export.

  Strong rules kill on their own: nothing to read, a physically impossible
  chars/sec, a tiled chant, a repeat of the line before. Everything else needs
  JUNK_MIN_SIGNALS weak signals to agree, so a real short lyric line (嗚呼, Oh,
  ラララ) survives on being short alone."""
  start, end, text = seg[0], seg[1], _norm(seg[2])
  if not any(c.isalnum() for c in text):
    return True
  if len(text) / max(end - start, MIN_DUR) >= JUNK_CPS_HI:
    return True
  if len(text) >= JUNK_MIN_CHARS and _repeated_unit(text):
    return True
  if _near_repeat(text, _norm(prev_text)):
    return True
  return len(junk_signals(seg)) >= JUNK_MIN_SIGNALS

def filter_junk(segments):
  """Drop junk cues, threading the previous kept text through the repeat rule."""
  kept = []
  prev = None
  for seg in segments or []:
    if is_junk_cue(seg, prev):
      continue
    kept.append(seg)
    prev = seg[2]
  return kept

def mc_score(segment):
  """Weighted MC score for one ASR segment (start, end, text). Higher = more
  likely spoken. Each signal caps at one hit's worth, so a single chatty
  line can't outvote everything else."""
  start, end, text = segment[0], segment[1], _norm(segment[2])
  if len(text) < MIN_CHARS:
    return 0.0
  score = 0.0
  punct = sum(1 for c in text if c in PUNCT_CHARS)
  if punct / len(text) >= PUNCT_DENSITY_HI:
    score += W_PUNCT
  if _hits(text, _POLITE):
    score += W_POLITE
  if _hits(text, _FILLER):
    score += W_FILLER
  dur = max(end - start, MIN_DUR)
  if len(text) / dur >= RATE_FAST:
    score += W_RATE
  return score

def classify_mc(segments):
  """Per-segment MC flags for an ASR segment list."""
  return [mc_score(seg) >= MC_THRESHOLD for seg in segments or []]

def mc_spans(segments, flags=None, gap=SPAN_GAP):
  """Merge adjacent MC segments into [(start, end), ...], joining gaps under
  `gap` seconds so one spoken block is one span."""
  segments = segments or []
  if flags is None:
    flags = classify_mc(segments)
  spans = []
  for (s, e, _), is_mc in zip(segments, flags):
    if not is_mc:
      continue
    if spans and s - spans[-1][1] <= gap:
      spans[-1] = (spans[-1][0], max(spans[-1][1], e))
    else:
      spans.append((s, e))
  return spans

def in_spans(start, end, spans):
  """True when a segment overlaps any span."""
  return any(start < b and end > a for a, b in spans or [])

def strip_mc(segments, spans):
  """Drop segments overlapping an MC span, so song matching never sees patter."""
  if not spans:
    return list(segments or [])
  return [seg for seg in segments or [] if not in_spans(seg[0], seg[1], spans)]

def mc_cues(segments, flags=None):
  """MC segments as their own cues, text verbatim from ASR.
  Cue.confidence = 'mc' tags them for export styles."""
  from .align import Cue
  segments = segments or []
  if flags is None:
    flags = classify_mc(segments)
  return [Cue(s, e, t, "mc") for (s, e, t), is_mc in zip(segments, flags) if is_mc]

def region_mc_frac(segments, flags, region):
  """Share of the region's ASR segments flagged MC (0.0 when it has none)."""
  pairs = [f for (s, e, _), f in zip(segments, flags)
           if s >= region.start - 0.5 and e <= region.end + 0.5]
  if not pairs:
    return 0.0
  return sum(pairs) / len(pairs)

def mark_mc_regions(regions, segments, flags=None, region_scored=None,
                    threshold=None, majority=MC_MAJORITY):
  """Set region.mc on regions mostly spoken. Returns flagged count.

  Two rules, both required when scores known: majority-MC segments, and no
  candidate clearing `threshold`. A real song whose ASR came through
  punctuated still matches a lyric, so the score vetoes the flag."""
  if threshold is None:
    from .align import SIMILARITY_THRESHOLD
    threshold = SIMILARITY_THRESHOLD
  if flags is None:
    flags = classify_mc(segments)
  n = 0
  for i, region in enumerate(regions):
    if region_mc_frac(segments, flags, region) < majority:
      continue
    scored = (region_scored or [])[i] if region_scored and i < len(region_scored) else []
    if scored and max(s for s, _ in scored) >= threshold:
      continue
    region.mc = True
    n += 1
  return n
