"""Configurable dir resolution (QSettings > env > default) and size formatting."""
from pathlib import Path

from utasub.core import paths
from utasub.gui.util import human_size

def test_human_size_units():
  assert human_size(0) == "0 B"
  assert human_size(512) == "512 B"
  assert human_size(1024) == "1.0 KB"
  assert human_size(1536) == "1.5 KB"
  assert human_size(432013312) == "412.0 MB"
  assert human_size(3 * 1024 ** 3) == "3.0 GB"

def test_resolve_dir_order(monkeypatch):
  monkeypatch.setattr(paths, "setting", lambda key: "")  # no Qt, no stored value
  monkeypatch.delenv("PATHS_TEST_DIR", raising=False)
  default = Path("/default/dir")

  assert paths.resolve_dir("paths/test", "PATHS_TEST_DIR", default) == default

  monkeypatch.setenv("PATHS_TEST_DIR", "/from/env")
  assert paths.resolve_dir("paths/test", "PATHS_TEST_DIR", default) == Path("/from/env")

  monkeypatch.setattr(paths, "setting", lambda key: "/from/settings")
  assert paths.resolve_dir("paths/test", "PATHS_TEST_DIR", default) == Path("/from/settings")

  # no env var configured for a row: settings, then default
  monkeypatch.setattr(paths, "setting", lambda key: "")
  assert paths.resolve_dir("paths/test", None, default) == default
