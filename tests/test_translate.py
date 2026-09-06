"""LLM translate: numbered-response parsing, chunking, request body. No network:
urllib.request.urlopen is monkeypatched."""
import json

from utasub.core import translate

class _FakeResp:
  def __init__(self, payload):
    self._payload = payload
    self.status = 200
  def read(self):
    return self._payload
  def __enter__(self):
    return self
  def __exit__(self, *a):
    return False

def _chat_payload(content):
  return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")

def test_parse_maps_numbers_and_fills_missing():
  out = translate._parse("1. hello\n3. world", 3)
  assert out == ["hello", "", "world"]

def test_parse_drops_extra_and_out_of_range_numbers():
  out = translate._parse("1. a\n2. b\n5. z\nnoise", 2)
  assert out == ["a", "b"]

def test_parse_accepts_paren_form():
  assert translate._parse("1) x\n2) y", 2) == ["x", "y"]

def test_translate_lines_chunks_and_preserves_length(monkeypatch):
  monkeypatch.setattr(translate, "CHUNK", 2)
  bodies = []
  def fake_urlopen(req, timeout=None):
    bodies.append(json.loads(req.data.decode("utf-8")))
    # echo each input line number back as "T<n>"
    lines = bodies[-1]["messages"][1]["content"].splitlines()
    reply = "\n".join(f"{i + 1}. T{i + 1}" for i in range(len(lines)))
    return _FakeResp(_chat_payload(reply))
  monkeypatch.setattr(translate.urllib.request, "urlopen", fake_urlopen)
  out = translate.translate_lines(["a", "b", "c"], "en", "http://x/v1")
  assert out == ["T1", "T2", "T1"]     # per-chunk numbering restarts at 1
  assert len(bodies) == 2              # 3 lines / chunk 2 -> two requests
  assert bodies[0]["messages"][1]["content"] == "1. a\n2. b"
  assert bodies[1]["messages"][1]["content"] == "1. c"

def test_body_omits_model_when_blank_and_sends_it_otherwise(monkeypatch):
  seen = []
  def fake_urlopen(req, timeout=None):
    seen.append(json.loads(req.data.decode("utf-8")))
    return _FakeResp(_chat_payload("1. x"))
  monkeypatch.setattr(translate.urllib.request, "urlopen", fake_urlopen)
  translate.translate_lines(["a"], "en", "http://x/v1")
  assert "model" not in seen[-1]
  translate.translate_lines(["a"], "en", "http://x/v1", model="qwen")
  assert seen[-1]["model"] == "qwen"

def test_endpoint_alive(monkeypatch):
  monkeypatch.setattr(translate.urllib.request, "urlopen",
                      lambda url, timeout=None: _FakeResp(b"{}"))
  assert translate.endpoint_alive("http://x/v1") is True
  def boom(url, timeout=None):
    raise OSError("refused")
  monkeypatch.setattr(translate.urllib.request, "urlopen", boom)
  assert translate.endpoint_alive("http://x/v1") is False
