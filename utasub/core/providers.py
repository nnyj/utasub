"""Lyric candidate fetch: shared providers from lyrickit, plus utasub-specific
sources (embedded tags, curated library).
Order: embedded tags, NetEase primary, LRCLib fallback."""
import re
from pathlib import Path

# re-exported for the rest of utasub (session, setlist, gui, cli import here)
from lyrickit import (  # noqa: F401
  LYRICS_DIR, Candidate,
  build_queries, clean_title, fetch_from_providers, lrclib_candidates,
  media_tags, netease_candidates, parse_lrc,
)

def embedded_lyrics(path, tags=None):
  """Lyrics tag from container metadata. Returns Candidate or None.
  tags: precomputed media_tags dict (avoids re-calling ffprobe)."""
  if tags is None:
    tags = media_tags(path)
  lrc = tags.get("lyrics") or tags.get("unsyncedlyrics")
  if not lrc or not lrc.strip():
    return None
  return Candidate(
    title=tags.get("title", Path(path).stem),
    album=tags.get("album", ""),
    artist=tags.get("artist", ""),
    source="Metadata tags",
    duration_ms=tags.get("_duration_ms", 0),
    lrc=lrc,
  )

# --- translations ---

TRANSLATION_NEAR_S = 0.5   # a translated line this close to an LRC line is that line

def translation_lines(candidate):
  """{original line text: translated text} from a candidate's `tlyric`.
  Translated lines carry their own timestamps, so they match the LRC by time
  within TRANSLATION_NEAR_S and fall back to position when a provider timed the
  translation off its own copy of the song. Keyed by text, since a cue reaches
  the exporter with its lyric line but not the LRC time it came from."""
  tlyric = getattr(candidate, "tlyric", "")
  if not tlyric or not candidate.lrc:
    return {}
  orig, trans = parse_lrc(candidate.lrc), parse_lrc(tlyric)
  out, used = {}, set()
  for i, (t, text) in enumerate(orig):
    near = [j for j, (tt, _) in enumerate(trans)
            if tt is not None and j not in used and abs(tt - t) <= TRANSLATION_NEAR_S
            ] if t is not None else []
    hit = min(near, key=lambda j: abs(trans[j][0] - t)) if near else (
      i if i < len(trans) and i not in used else None)
    if hit is None:
      continue
    used.add(hit)
    if trans[hit][1] != text:
      out[text] = trans[hit][1]
  return out

# --- field alternates (picker prefill, artist hint) ---

def _parse_filename(path):
  """Extract Artist, Title from 'Artist - Title' filename patterns."""
  stem = Path(path).stem
  # strip youtube id suffix [xxxx], date tags [20xxxxxx], parentheticals
  stem = re.sub(r"\s*\[[\w-]+\]\s*$", "", stem)
  stem = re.sub(r"\s*\[\d{8}\]\s*$", "", stem)
  # fullwidth/unicode quotes → strip
  stem = re.sub(r'["""＂「」『』【】]', "", stem)
  stem = re.sub(r"\bfrom\b.*", "", stem, flags=re.IGNORECASE).strip()
  if " - " in stem:
    parts = stem.split(" - ", 1)
    return {"artist": parts[0].strip(), "title": parts[1].strip()}
  return {"title": stem.strip()}

def build_alternates(media_path, tags=None):
  """Build dict of field → list of (value, source_hint) alternates.
  First entry in each list = prefilled default. Artist comes from container tags,
  then 'Artist - Title' filename; parent folder is unreliable, so never used."""
  alts = {"title": [], "album": [], "artist": []}
  seen = {k: set() for k in alts}

  def add(field, val, src):
    v = val.strip()
    if not v or v in seen[field]:
      return
    seen[field].add(v)
    alts[field].append((v, src))

  # container tags first (primary prefill), cleaned song name before raw tag
  if tags:
    raw_title = tags.get("title", "")
    add("title", clean_title(raw_title), "tags")
    add("title", raw_title, "tags raw")
    add("album", tags.get("album", ""), "tags")
    for a in tags.get("artist", "").split("/"):
      add("artist", a, "tags")

  fp = _parse_filename(media_path)
  add("title", fp.get("title", ""), "filename")
  add("artist", fp.get("artist", ""), "filename")

  for k in alts:
    if not alts[k]:
      alts[k].append(("", ""))
  return alts

# --- curated library ---

def _sanitize_filename(name):
  return re.sub(r'[<>:"/\\|?*]', "_", name).strip()

def curated_lrc(title, artists):
  """Check curated library for an existing .lrc file. Returns text or None."""
  if not LYRICS_DIR.exists():
    return None
  files = [LYRICS_DIR / f"{_sanitize_filename(n)}.lrc"
           for n in [f"{a} - {title}" for a in artists] + [title]]
  cached = next((f for f in files if f.exists()), None)
  if cached:
    text = cached.read_text(encoding="utf-8")
    return text if text.strip() else None
  return None

# --- top-level fetch ---

def fetch_candidates(path, providers=None, fresh=False, tags=None):
  """Fetch all lyric candidates for a media file.
  Returns Candidate list, embedded first then per-provider results.
  providers: list of str, default ["NetEase"].
  tags: precomputed media_tags dict (avoids re-calling ffprobe)."""
  if providers is None:
    providers = ["NetEase"]
  if tags is None:
    tags = media_tags(path)
  candidates = []

  emb = embedded_lyrics(path, tags=tags)
  if emb:
    candidates.append(emb)

  queries, title, artists = build_queries(path, tags=tags)

  curated = curated_lrc(title, artists)
  if curated:
    candidates.append(Candidate(
      title=title, album=tags.get("album", ""),
      artist="/".join(artists) if artists else "",
      source="Curated", duration_ms=tags.get("_duration_ms", 0),
      lrc=curated,
    ))

  for query in queries:
    print(f"  lyrics search: {query} [{','.join(providers)}]")
    candidates.extend(fetch_from_providers(query, providers, fresh))
    if any(c.lrc for c in candidates):
      break  # got results, skip other queries

  return candidates
