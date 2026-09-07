"""Local LLM translation over an OpenAI-compatible chat endpoint (llama-server).
Provider-neutral: one POST per batch of lines, numbered in and out so a cue keeps
its slot even when a line comes back empty. No third-party SDK, urllib only."""
import json
import re
import urllib.error
import urllib.request

ENDPOINT_DEFAULT = "http://localhost:8080/v1"
MODEL_DEFAULT = ""
TARGET_DEFAULT = "en"
CHUNK = 40                       # lines per request, keeps the prompt small
TIMEOUT_S = 120                  # a long batch on a local model can be slow
_NUM = re.compile(r"^\s*(\d+)[.)]\s*(.*)$")

def _system_prompt(target):
  return (
    f"Translate song lyrics and concert stage speech to {target}. "
    "Input is numbered lines `N. text`. Output the same count of numbered lines "
    "`N. translation`, one per input line, numbers matching. Translate the "
    "meaning naturally. No commentary, no blank lines, no extra numbers.")

def _post(endpoint, body):
  req = urllib.request.Request(
    f"{endpoint.rstrip('/')}/chat/completions",
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"})
  with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
    data = json.loads(resp.read().decode("utf-8"))
  return data["choices"][0]["message"]["content"]

def _parse(content, n):
  """Numbered reply -> list of n strings, missing numbers left empty. Extra or
  out-of-range numbers are dropped, so a stray line never shifts the rest."""
  by_num = {}
  for line in content.splitlines():
    m = _NUM.match(line)
    if m:
      by_num[int(m.group(1))] = m.group(2).strip()
  return [by_num.get(i + 1, "") for i in range(n)]

def translate_lines(texts, target, endpoint, model=""):
  """Translate `texts` to `target`, returning a list the same length (empty
  string where the model gave nothing for that line). Batched in CHUNK lines."""
  out = []
  for i in range(0, len(texts), CHUNK):
    chunk = texts[i:i + CHUNK]
    numbered = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(chunk))
    body = {
      "messages": [
        {"role": "system", "content": _system_prompt(target)},
        {"role": "user", "content": numbered},
      ],
      "temperature": 0,
      "stream": False,
    }
    if model:  # blank omitted: llama-server serves its single loaded model
      body["model"] = model
    out.extend(_parse(_post(endpoint, body), len(chunk)))
  return out

def endpoint_alive(endpoint):
  """True when the endpoint answers GET /models within 2s."""
  try:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/models", timeout=2) as r:
      return r.status == 200
  except (OSError, urllib.error.URLError, ValueError) as e:
    print(f"  translate endpoint down: {endpoint}: {e}")
    return False
