"""Configurable cache/data directories: QSettings override, env var, default.

Qt is optional: the CLI runs without PySide6, so the QSettings read is guarded
and the resolution simply falls through to env/default when Qt is missing."""
import os
import tempfile
from pathlib import Path

ORG = APP = "utasub"

# key in QSettings -> env var lyrickit / huggingface read at import time
ENV_KEYS = (("paths/lyrics_dir", "LYRICS_DIR"), ("paths/hf_home", "HF_HOME"))

def default_stem_cache():
  base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
  return Path(base) / "utasub" / "stems"

def default_lyrics_dir():
  return Path.home() / ".cache" / "lyrickit"

def default_hf_home():
  return Path.home() / ".cache" / "huggingface"

def setting(key):
  """Stored override for `key`, "" when unset or Qt unavailable."""
  try:
    from PySide6.QtCore import QSettings
  except ImportError:
    return ""
  return (QSettings(ORG, APP).value(key, "", str) or "").strip()

def resolve_dir(key, env, default):
  """QSettings `key`, else $`env`, else `default`. No filesystem touch."""
  val = setting(key)
  if not val and env:
    val = (os.environ.get(env) or "").strip()
  return Path(val or default)

def cache_dir(key, env, default):
  """resolve_dir, created on demand."""
  d = resolve_dir(key, env, default)
  d.mkdir(parents=True, exist_ok=True)
  return d

def apply_path_env():
  """Push the stored lyrics / HF paths into os.environ. lyrickit and
  huggingface bind their dir once at import, so this must run before either is
  imported (cli.main calls it on the GUI path only, CLI keeps env as given)."""
  for key, env in ENV_KEYS:
    val = setting(key)
    if val:
      os.environ[env] = val
