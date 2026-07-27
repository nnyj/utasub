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
