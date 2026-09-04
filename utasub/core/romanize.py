"""Shared romanization + script detection for CJK, thin layer over romakit.

Adds what romakit has no opinion on: user/song-level locale pin, and a
warning when a romakit language extra is missing from the environment.
"""
import unicodedata

from romakit import LANGS, detect, is_cjk, norm_lang, romanize as _rk_romanize

def nfkc(text):
  """Fold compatibility characters to their plain forms: the ﬁ/ﬂ ligatures a
  lyric source may carry become fi/fl, full-width latin folds to ascii, so the
  MMS dictionary times them and karaoke does not orphan the leading glyph."""
  return unicodedata.normalize("NFKC", text) if text else text

_DEP_HINT = {"zh": "pypinyin not installed, no Chinese romanization",
             "ko": "korean-romanizer not installed, no Korean romanization"}
_warned = set()

def _romanize_lang(text, loc):
  """romakit dispatch for one locale. '' (+ one warning) if the romanizer is
  missing: the base dependency is romakit[jp,zh,ko], so this is a broken env,
  and an empty key scores every candidate alike downstream."""
  try:
    return _rk_romanize(text, lang=loc)
  except ImportError:
    if loc not in _warned:
      _warned.add(loc)
      print(f"  ! {_DEP_HINT[loc]} (reinstall romakit[jp,zh,ko])")
    return ""

_lang_override = ""
_default_locale = ""

def set_lang_override(lang):
  """Pin lyric language from UI/CLI ('auto'|'jp'|'cn'|'kr').
  Beats both song-level default and any explicit locale= arg, since those
  come from auto-detection; 'auto' clears it."""
  global _lang_override
  _lang_override = norm_lang(lang)

def get_lang_code():
  """Pinned UI code ('jp'|'cn'|'kr'), '' when auto. Lets GUI show what
  CLI set."""
  return {"ja": "jp", "zh": "cn", "ko": "kr"}.get(_lang_override, "")

def set_default_locale(locale):
  """Pin locale for subsequent romanize() calls without an explicit one.
  Songs are single-language, so callers detect once over whole lyric and pin
  it here; per-line detection would misread kanji-only Japanese as zh.
  Pass '' to clear."""
  global _default_locale
  _default_locale = locale or ""

def get_default_locale():
  return _default_locale

def romanize(text, locale=None):
  """Romanize CJK text; non-CJK passes through.
  Locale resolution: user override > explicit arg > module default > detect().
  locale='' is explicit "not CJK", distinct from None (unset)."""
  loc = locale if locale is not None else _default_locale
  guessed = not loc
  if guessed and locale is None:
    loc = detect(text)
  if _lang_override:
    loc, guessed = _lang_override, False
  if loc not in ("zh", "ko"):
    return nfkc(_rk_romanize(text, lang="jp"))
  rom = _romanize_lang(text, loc)
  # safety net only when nothing pinned locale: han-only Japanese lyrics
  # detect as zh, fall back to kanji path if pypinyin absent
  if not rom and loc == "zh" and guessed:
    return nfkc(_rk_romanize(text, lang="jp"))
  return nfkc(rom)

def mora_units(text):
  """Japanese romaji split into karaoke units, one per kana mora (foreign words
  whole). () when text has no Japanese. Chinese/Korean need no split, their
  romaji is already one syllable per token."""
  from romakit import mora
  return mora(nfkc(text))

def cjk_pairs(text, locale=None):
  """[(glyph/word surface, romaji)] for per-unit karaoke on the original line:
  a word for Japanese, a glyph for Chinese/Korean. () for non-CJK. Locale
  resolves as romanize() does (override > arg > default)."""
  from romakit import pairs as _pairs
  loc = _lang_override or (locale if locale is not None else _default_locale)
  return _pairs(nfkc(text), lang=loc or "auto")

def romanize_suffix(text):
  """' (Romaji)' suffix if text is CJK, else empty string."""
  if not is_cjk(text):
    return ""
  rom = romanize(text)
  if not rom or rom == text:
    return ""
  return f" ({rom})"
