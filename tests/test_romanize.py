"""Script detection + CJK romanize dispatch, incl. graceful degrade when
optional zh/ko packages absent."""
import builtins
import romakit
import sys

import pytest

from utasub.core import romanize as rz
from utasub.core.export import srt_name

@pytest.fixture(autouse=True)
def _reset_pins():
  """Prevent cross-test leak of module-level song locale / lang override."""
  yield
  rz.set_default_locale("")
  rz.set_lang_override("auto")

def test_detect_priority():
  assert rz.detect("君の名は") == "ja"       # kana wins
  assert rz.detect("你好世界") == "zh"       # han only
  assert rz.detect("안녕하세요") == "ko"     # hangul
  assert rz.detect("한국어 mixed 漢字") == "ko"  # hangul beats han
  assert rz.detect("hello world") == ""
  assert rz.detect("") == ""

def test_is_cjk():
  assert rz.is_cjk("カタカナ")
  assert rz.is_cjk("中文")
  assert rz.is_cjk("한글")
  assert not rz.is_cjk("plain ascii 123")

def test_romanize_japanese():
  assert rz.romanize("こんにちは").lower().startswith("kon")
  assert rz.romanize("hello") == "hello"

def _block_import(monkeypatch, dep, loc):
  """Force `import dep` to fail, drop cached romakit submodule + warn flag."""
  real = builtins.__import__

  def fake(mod, *a, **kw):
    if mod == dep or mod.startswith(dep + "."):
      raise ImportError(mod)
    return real(mod, *a, **kw)
  monkeypatch.setattr(builtins, "__import__", fake)
  monkeypatch.delitem(sys.modules, f"romakit.{loc}", raising=False)
  # `from . import ko` reads the attribute off the package when it is still set,
  # so dropping the sys.modules entry alone leaves an earlier import in place
  monkeypatch.delattr(romakit, loc, raising=False)
  rz._warned.discard(loc)

@pytest.mark.parametrize("dep,loc,text", [("pypinyin", "zh", "你好"),
                                          ("korean_romanizer", "ko", "안녕하세요")])
def test_romanize_missing_dep_returns_empty(monkeypatch, dep, loc, text):
  _block_import(monkeypatch, dep, loc)
  assert rz.romanize(text, locale=loc) == ""

def test_romanize_han_only_falls_back_to_ja(monkeypatch):
  """Auto-detected han-only text keeps Japanese path when pypinyin absent
  (JP lyrics often kanji-only)."""
  _block_import(monkeypatch, "pypinyin", "zh")
  assert rz.romanize("漢字") not in ("", "漢字")

def test_default_locale_forces_ja_for_han_only():
  """zh delegation works; song-level pin keeps kanji-only lines on Japanese
  path (auto-detect would call them zh)."""
  pytest.importorskip("pypinyin")
  assert rz.romanize("漢字") == "han zi"  # auto-detect: zh
  rz.set_default_locale("ja")
  assert rz.romanize("漢字").lower().startswith("kanji")

def test_default_locale_yields_to_explicit_arg():
  rz.set_default_locale("ja")
  assert rz.get_default_locale() == "ja"
  pytest.importorskip("pypinyin")
  assert rz.romanize("你好", locale="zh") == "ni hao"

def test_explicit_empty_locale_ignores_default():
  """locale='' means explicitly not-CJK; pinned default can't override."""
  rz.set_default_locale("zh")
  assert rz.romanize("hello", locale="") == "hello"

def test_srt_name_locale_suffix(tmp_path):
  media = tmp_path / "v.mp4"
  assert srt_name(media, [(0, 1, "こんにちは")]).name == "v.ja.srt"
  assert srt_name(media, [(0, 1, "안녕")]).name == "v.ko.srt"
  assert srt_name(media, [(0, 1, "你好")]).name == "v.zh.srt"
  assert srt_name(media, [(0, 1, "hello")]).name == "v.srt"

def test_lang_override_beats_detect_and_pin():
  pytest.importorskip("pypinyin")
  rz.set_default_locale("ja")
  rz.set_lang_override("cn")
  assert rz.romanize("漢字") == "han zi"                # beats song pin
  assert rz.romanize("漢字", locale="ja") == "han zi"   # beats explicit arg

def test_lang_code_roundtrip_and_auto_clears():
  """GUI reads back UI code, so CLI --lang shows checked in menu."""
  rz.set_lang_override("cn")
  assert rz.get_lang_code() == "cn"
  rz.set_lang_override("auto")
  assert rz.get_lang_code() == ""
  assert rz.romanize("こんにちは").lower().startswith("kon")
