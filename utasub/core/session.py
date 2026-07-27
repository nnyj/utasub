"""Session file <name>.utasub.json next to media: the cached ASR block plus
candidates, per-region choices, toggles, regions, manual timings.
Deterministic re-export."""
import json
import re
from dataclasses import asdict, fields as dataclass_fields
from statistics import median
from pathlib import Path

from .align import Cue
from .providers import Candidate

SCHEMA_VERSION = 1


def session_path(media_path):
  """<name>.utasub.json next to media file."""
  return Path(media_path).with_suffix(".utasub.json")


# --- manual edits ---
# A re-align rebuilds every cue from scratch, so an edit must find its line
# again afterwards. Position alone breaks on insert (renumbers the rest); text
# alone breaks on repeats (chorus). So an edit keys on text plus the position
# it was made at, re-attaching to the nearest same-text cue. Wrong only when
# repeated text moves further than the gap to its neighbouring repeat.

def edit_key(text):
  """Normalized cue text, the stable half of an edit's identity."""
  return re.sub(r"\s+", " ", (text or "").strip().lower())


def _edit_record(cue, start=None, end=None):
  return {"text": edit_key(cue[2]), "at": round(cue[0], 3),
          "start": round(cue[0] if start is None else start, 3),
          "end": round(cue[1] if end is None else end, 3)}


def apply_manual_edits(cues, edits):
  """Re-apply manual timings and deletions to a freshly built cue list.
  Unmatched edits stay in the record: a cue missing from this pass isn't
  evidence the edit is stale, a later align may still find its line.
  Returns a new list."""
  timings = (edits or {}).get("timings") or []
  deleted = (edits or {}).get("deleted") or []
  if not timings and not deleted:
    return list(cues)

  by_text = {}
  for i, c in enumerate(cues):
    by_text.setdefault(edit_key(c[2]), []).append(i)
  used = set()

  def claim(rec):
    free = [i for i in by_text.get(rec.get("text", ""), []) if i not in used]
    if not free:
      return None
    at = rec.get("at")
    i = min(free, key=lambda j: abs(cues[j][0] - at)) if at is not None else free[0]
    used.add(i)
    return i

  out = list(cues)
  drop = set()
  for rec in deleted:            # deletions claim first: suppressed cue is
    i = claim(rec)               # gone regardless of timing
    if i is not None:
      drop.add(i)
  for rec in timings:
    i = claim(rec)
    if i is None:
      continue
    c = out[i]
    if isinstance(c, Cue):
      out[i] = Cue(rec["start"], rec["end"], c.text, c.confidence)
    else:
      out[i] = (rec["start"], rec["end"]) + tuple(c[2:])
  return [c for i, c in enumerate(out) if i not in drop]


def _migrate_manual_timings(data):
  """Convert position-keyed manual timings from an old session to text-keyed.
  Positions index one of two lists (lyric cues alone, or every cue including
  raw ASR between songs) depending on which screen wrote them; both are tried
  and the one whose times land on the cues they claim wins."""
  cues = data.get("cues") or []
  lyric = [c for c in cues if len(c) > 3 and isinstance(c[3], str)]
  raw = (data.get("manual_timings") or {}).items()
  pairs = []
  for k, v in raw:
    try:
      i = int(k)
    except (TypeError, ValueError):
      continue
    if isinstance(v, (list, tuple)) and len(v) >= 2:
      pairs.append((i, float(v[0]), float(v[1])))
  if not pairs:
    return {"timings": [], "deleted": []}

  def score(seq):
    d = [abs(s - seq[i][0]) for i, s, _ in pairs if 0 <= i < len(seq)]
    return (median(d) if d else float("inf"), -len(d))

  seq = min((cues, lyric), key=score)
  out = [{"text": edit_key(seq[i][2]), "at": s, "start": s, "end": e}
         for i, s, e in pairs if 0 <= i < len(seq)]
  return {"timings": out, "deleted": []}


def _cand_dict(c):
  """Candidate → plain dict, dropping the cached parse."""
  d = asdict(c)
  d.pop("_parsed_lines", None)
  return d


def _cand_from(d):
  """Plain dict → Candidate, ignoring keys Candidate does not declare."""
  fields = {f.name for f in dataclass_fields(Candidate)}
  return Candidate(**{k: v for k, v in d.items() if k in fields})


def _read_raw(media_path):
  """Parsed session file dict, or None when the file is missing."""
  path = session_path(media_path)
  if not path.exists():
    return None
  return json.loads(path.read_text(encoding="utf-8"))


def _write_raw(media_path, data):
  path = session_path(media_path)
  path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
  return path


# --- ASR block (written by asr_gen, read by core.asr.load_asr) ---

def save_asr(media_path, language, segments):
  """Write/update the ASR block, preserving any existing session fields.
  segments = [(start, end, text), ...]. Returns the session path."""
  data = _read_raw(media_path) or {
    "schema_version": SCHEMA_VERSION,
    "media": str(Path(media_path).resolve()),
  }
  data["asr"] = {
    "language": language,
    "segments": [[round(s, 3), round(e, 3), t] for s, e, t in segments],
  }
  return _write_raw(media_path, data)


def has_asr(media_path):
  """True when the file carries a cached ASR block."""
  data = _read_raw(media_path)
  return bool(data and data.get("asr"))


def has_session(media_path):
  """True when saved editing state (cues) exists, not just an ASR block."""
  data = _read_raw(media_path)
  return bool(data and "cues" in data)


def save(media_path, *, candidates=None, chosen_index=None, cues=None,
         song_span=None, romaji=True, credit_toggles=None,
         regions=None, placement=None, region_metas=None,
         offset=None, lead=None, region_choices=None, manual_edits=None,
         mc_spans=None):
  """Write session file. candidates = list of Candidate dataclass.
  region_metas: per-region strategy dicts from multi-song finalize.
  region_choices: {region_idx: (score, Candidate)} picked per region, so
  reopen restores picker assignments without re-searching.
  mc_spans: [(start, end), ...] detected spoken intermissions, stored so
  re-export keeps the MC track without re-classifying.
  offset/lead: SRT write-time timing, stored so re-export reproduces the SRT."""
  from .export import SRT_OFFSET_S, SRT_LEAD_S
  data = {
    "schema_version": SCHEMA_VERSION,
    "media": str(Path(media_path).resolve()),
    "chosen_index": chosen_index,
    "candidates": [_cand_dict(c) for c in (candidates or [])],
    "cues": [list(c) if isinstance(c, (tuple, Cue)) else c for c in (cues or [])],
    "song_span": list(song_span) if song_span else None,
    "romaji": romaji,
    "credit_toggles": credit_toggles or {},
    "regions": regions or [],
    "manual_edits": manual_edits or {"timings": [], "deleted": []},
    "placement": placement or {},
    "region_metas": region_metas or [],
    "mc_spans": [[round(a, 3), round(b, 3)] for a, b in (mc_spans or [])],
    "offset": SRT_OFFSET_S if offset is None else offset,
    "lead": SRT_LEAD_S if lead is None else lead,
    "region_choices": [
      {"region_idx": idx, "score": score, "candidate": _cand_dict(cand)}
      for idx, (score, cand) in sorted((region_choices or {}).items())
    ],
  }
  # keep the ASR block; session edits never touch it
  existing = _read_raw(media_path)
  if existing and "asr" in existing:
    data["asr"] = existing["asr"]
  return _write_raw(media_path, data)


def load(media_path):
  """Load session file. Returns dict or None if missing."""
  path = session_path(media_path)
  if not path.exists():
    return None
  data = json.loads(path.read_text(encoding="utf-8"))
  data["candidates"] = [_cand_from(c) for c in data.get("candidates", [])]
  data["region_choices"] = {
    ch["region_idx"]: (ch.get("score", 0.0), _cand_from(ch["candidate"]))
    for ch in data.get("region_choices", [])
  }
  # cues: reconstruct Cue for song lines, keep tuples for ASR segments
  raw_cues = []
  for c in data.get("cues", []):
    if len(c) == 4 and isinstance(c[3], str):
      raw_cues.append(Cue(c[0], c[1], c[2], c[3]))
    else:
      raw_cues.append(tuple(c))
  data["cues"] = raw_cues
  data["song_span"] = tuple(data["song_span"]) if data.get("song_span") else None
  data["mc_spans"] = [tuple(s) for s in data.get("mc_spans", [])]
  # best-effort migration: old session carries only positional timings.
  # File itself untouched until next save, which writes both forms.
  edits = data.get("manual_edits")
  if not (edits or {}).get("timings") and data.get("manual_timings"):
    edits = _migrate_manual_timings(data)
  data["manual_edits"] = edits or {"timings": [], "deleted": []}
  return data


def reexport(media_path, session_data):
  """Deterministic re-export from session data. No network, no re-align.
  Multi-region sessions export from pre-merged cues (stored time-ordered).
  Returns list of written paths."""
  from .export import export_srt
  cues = session_data["cues"]
  romaji = session_data.get("romaji", True)
  return export_srt(media_path, cues, romaji=romaji,
                    offset=session_data.get("offset"),
                    lead=session_data.get("lead"))
