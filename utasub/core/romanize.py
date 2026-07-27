"""Shared romanization + script detection for CJK, thin layer over romakit.

Adds what romakit has no opinion on: user/song-level locale pin, and a
warning when optional zh (`pypinyin`) / ko (`korean-romanizer`) extras
are missing.
"""
from romakit import LANGS, detect, is_cjk, norm_lang, romanize as _rk_romanize

_DEP_HINT = {"zh": "pypinyin not installed, skipping Chinese romanization"
                   " (pip install 'utasub[zh]')",
             "ko": "korean-romanizer not installed, skipping Korean"
                   " romanization (pip install 'utasub[ko]')"}
_warned = set()


def _romanize_lang(text, loc):
  """romakit dispatch for one locale. '' (+ one warning) if extra missing."""
  try:
    return _rk_romanize(text, lang=loc)
  except ImportError:
    if loc not in _warned:
      _warned.add(loc)
      print(f"  ! {_DEP_HINT[loc]}")
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
    return _rk_romanize(text, lang="jp")
  rom = _romanize_lang(text, loc)
  # safety net only when nothing pinned locale: han-only Japanese lyrics
  # detect as zh, fall back to kanji path if pypinyin absent
  if not rom and loc == "zh" and guessed:
    return _rk_romanize(text, lang="jp")
  return rom


def romanize_suffix(text):
  """' (Romaji)' suffix if text is CJK, else empty string."""
  if not is_cjk(text):
    return ""
  rom = romanize(text)
  if not rom or rom == text:
    return ""
  return f" ({rom})"
