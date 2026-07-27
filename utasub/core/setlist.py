"""Setlist discovery: Wikipedia API -> heuristic tracklist extraction.
Also parses a hand-pasted setlist (GUI dialog, --setlist-file)."""
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from .providers import LYRICS_DIR

TEXT_CAP = 15000  # max chars fed to extraction
USER_AGENT = "utasub/1.0 (setlist lookup)"


def _http_get(url, headers=None, timeout=10):
  """GET returning text (utf-8) or None on network failure."""
  req = urllib.request.Request(url, headers=headers or {"User-Agent": USER_AGENT})
  try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      raw = resp.read()
  except OSError:
    return None
  return raw.decode("utf-8", errors="replace")


def _slug(text):
  return re.sub(r'[<>:"/\\|?*\s]+', "_", text).strip("_").lower()[:80]


def _cache_write(path, data):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# --- Wikipedia ---

def _wiki_search(query, lang="ja", limit=3):
  """Search Wikipedia, return page titles."""
  url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search"
         f"&srsearch={urllib.parse.quote(query)}&format=json&srlimit={limit}")
  raw = _http_get(url)
  if not raw:
    return []
  data = json.loads(raw)
  return [r["title"] for r in data.get("query", {}).get("search", [])]


def _wiki_wikitext(title, lang="ja"):
  """Fetch wikitext for a page. Returns str or None."""
  url = (f"https://{lang}.wikipedia.org/w/api.php?action=parse"
         f"&page={urllib.parse.quote(title)}&prop=wikitext&format=json")
  raw = _http_get(url)
  if not raw:
    return None
  data = json.loads(raw)
  return data.get("parse", {}).get("wikitext", {}).get("*")


def _wiki_discover(artist, keywords):
  """Search ja + en Wikipedia for setlist/tracklist pages. Returns (source_label, text) list."""
  results = []
  for lang, kws in [
    ("ja", ["セットリスト", "ライブ", "収録曲"]),
    ("en", ["setlist", "track listing", "live"]),
  ]:
    for kw in kws:
      q = f"{artist} {' '.join(keywords)} {kw}"
      titles = _wiki_search(q, lang=lang)
      for t in titles:
        wt = _wiki_wikitext(t, lang=lang)
        if not wt:
          continue
        tracklist_kws = ["セットリスト", "収録曲", "track listing", "setlist",
                         "tracklist", "曲順", "曲目"]
        has_tracklist = any(k.lower() in wt.lower() for k in tracklist_kws)
        if has_tracklist:
          text = _extract_around_keywords(wt, tracklist_kws)
          results.append((f"Setlist:Wikipedia({lang}:{t})", text))
      if results:
        break  # found in this language, skip remaining keywords
  return results


def _extract_around_keywords(text, keywords, cap=TEXT_CAP):
  """Text around first matching keyword, capped."""
  lower = text.lower()
  for kw in keywords:
    idx = lower.find(kw.lower())
    if idx >= 0:
      start = max(0, idx - 200)
      return text[start:start + cap]
  return ""


# --- extraction ---

# numbered list patterns: "1. Title", "01. Title", "1 Title", "# Title" (ordered wiki), "M1. Title"
_RE_NUMBERED = re.compile(
  r"^[\s*#]*(?:M?\d{1,2})[.\s)）\]】]\s*(.+?)(?:\s*[\(（\[【].*)?$",
  re.MULTILINE
)

# wiki tracklist template: {{Track listing ... | title1 = X | title2 = Y ...}}
_RE_WIKI_TRACK = re.compile(
  r"\|\s*title(\d+)\s*=\s*(.+?)(?:\n|\||})",
  re.MULTILINE
)

def _extract_heuristic(text):
  """Extract tracklist from text using numbered-list heuristics.
  Returns list of (order, title) 1-indexed."""
  tracks = []
  wiki_matches = _RE_WIKI_TRACK.findall(text)
  if wiki_matches:
    for num_str, title in wiki_matches:
      title = title.strip().strip("'\"[]{}")
      title = re.sub(r"\[\[([^|\]]+)\|?[^\]]*\]\]", r"\1", title)  # [[link|text]] -> link
      if title:
        tracks.append((int(num_str), title))
    if tracks:
      tracks.sort(key=lambda x: x[0])
      return tracks

  matches = _RE_NUMBERED.findall(text)
  if len(matches) >= 3:
    return [(i + 1, m.strip()) for i, m in enumerate(matches)]

  return []


# --- pasted setlist ---

# header/section lines carrying no title
_RE_JUNK_LINE = re.compile(
  r"^(?:set\s?list|セットリスト|セトリ|tracklist|track\s?listing|曲目|曲順|"
  r"encore|アンコール|本編|w?encore\s*\d*)\s*[:：]?$", re.IGNORECASE)
# leading "Setlist:" on the same line as the first title
_RE_HEADER_PREFIX = re.compile(
  r"^(?:set\s?list|セットリスト|セトリ|tracklist|track\s?listing|曲目|曲順)\s*[:：]\s*",
  re.IGNORECASE)
_RE_BULLET = re.compile(r"^[-–—*•・·>]+\s*")
# "1. X", "01 X", "1) X", "[03] X", "M1. X"; ceiling: a title starting with a
# number plus space loses it (upgrade = require the number to be sequential)
_RE_LEAD_NUM = re.compile(
  r"^[\[【(（]?(?:M|EN)?\d{1,3}[\]】)）]?\s*(?:[.、,:：]\s*|\s+)", re.IGNORECASE)
_RE_TIMESTAMP = re.compile(r"[\[(]?\d{1,2}:\d{2}(?::\d{2})?[\])]?")
_RE_SEP = re.compile(r"\s*[,、/／｜|]\s*")


def _clean_pasted_line(line):
  """Strip bullets, numbering, timestamps, header prefixes off one pasted
  line. Returns title or ""."""
  s = _RE_HEADER_PREFIX.sub("", line.strip())
  s = _RE_TIMESTAMP.sub("", s, count=1).strip() if _RE_TIMESTAMP.match(s) else s
  s = _RE_BULLET.sub("", s).strip()
  s = _RE_LEAD_NUM.sub("", s).strip()
  s = re.sub(r"\s*[\[(]?\d{1,2}:\d{2}(?::\d{2})?[\])]?$", "", s).strip()
  s = s.strip("-–—・ \t")
  if not s or _RE_JUNK_LINE.match(s) or s.isdigit():
    return ""
  return s


def parse_pasted(text):
  """Parse a hand-pasted setlist into ordered titles.
  Tolerates one-per-line, "1. X"/"01 X"/"1) X", bullets, header/timestamp junk,
  and a single separator-joined line (comma, 、, /, ／, ｜, |).
  Dedupes, keeps first occurrence order."""
  if not text:
    return []
  lines = [t for t in (_clean_pasted_line(l) for l in text.splitlines()) if t]

  titles = []
  for line in lines:
    parts = [p for p in (_clean_pasted_line(p) for p in _RE_SEP.split(line)) if p]
    # split inline only when line is clearly a list, not a title with a comma
    if len(parts) >= 3 or (len(lines) == 1 and len(parts) >= 2):
      titles.extend(parts)
    else:
      titles.append(line)

  seen = set()
  out = []
  for t in titles:
    key = t.casefold()
    if key not in seen:
      seen.add(key)
      out.append(t)
  return out


# --- cache ---

def _cache_path(hints):
  """Cache file path from hints dict, readable "artist - keywords" name."""
  slug = _slug(f"{hints.get('artist', '')} - {' '.join(hints.get('keywords', []))}")
  return LYRICS_DIR / "setlist" / f"{slug}.json"


def _cache_read(hints, fresh=False):
  """Read cached setlist result. Returns dict or None."""
  if fresh:
    return None
  path = _cache_path(hints)
  if not path.exists():
    return None
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return None


def _cache_save(hints, data):
  """Save setlist result to cache."""
  _cache_write(_cache_path(hints), data)


# --- main API ---

def discover(hints, fresh=False):
  """Discover concert setlist from hints dict.
  hints = {artist: str, keywords: list[str]}
  Returns (order, title) list, source string, extraction method.
  Empty list = nothing found (cached too)."""
  artist = hints.get("artist", "")
  keywords = hints.get("keywords", [])
  if not artist:
    return [], "", "none"

  cached = _cache_read(hints, fresh)
  if cached is not None:
    return ([(t[0], t[1]) for t in cached.get("tracklist", [])],
            cached.get("source", ""),
            cached.get("method", "cached"))

  print(f"  setlist: searching for {artist} {' '.join(keywords)}...")
  all_sources = []

  try:
    all_sources.extend(_wiki_discover(artist, keywords))
  except Exception as e:
    print(f"  setlist: Wikipedia error: {e}")

  # first source that extracts anything wins
  for source_label, text in all_sources:
    tracklist = _extract_heuristic(text)
    if tracklist:
      result = {
        "tracklist": tracklist,
        "source": source_label,
        "method": "heuristic",
        "raw_text_len": len(text),
      }
      _cache_save(hints, result)
      print(f"  setlist: {len(tracklist)} tracks from {source_label} (heuristic)")
      return tracklist, source_label, "heuristic"

  # cache the empty result too, so a miss is not re-searched
  _cache_save(hints, {"tracklist": [], "source": "", "method": "none"})
  print(f"  setlist: no tracklist found ({len(all_sources)} pages checked)")
  return [], "", "none"


def hints_from_path(path, artist_hint=""):
  """Build hints dict from media file path and optional artist.
  Extracts keywords (tour/event names) from filename."""
  stem = Path(path).stem
  clean = re.sub(r"\[[\w-]+\]", "", stem)  # strip [id] suffixes
  clean = re.sub(r"\(.*?\)", "", clean)
  clean = re.sub(r"[_\-]+", " ", clean).strip()
  if artist_hint:
    clean = clean.replace(artist_hint, "").strip()
    for part in artist_hint.split():  # also strip common romanizations
      clean = clean.replace(part, "").strip()
  keywords = [w for w in clean.split() if len(w) >= 2
              and w.lower() not in {"live", "concert", "tour", "mv", "official",
                                    "full", "hd", "4k", "video", "audio"}]
  return {"artist": artist_hint, "keywords": keywords}
