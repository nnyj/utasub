"""Edit > Settings: where utasub keeps its stems, lyrics library and HF models.

A blank field means "use the env var, then the built-in default", shown as the
line edit's placeholder. Lyrics / HF paths only take effect next launch:
lyrickit and huggingface bind their dir at import time."""
from PySide6.QtWidgets import (
  QDialog, QDialogButtonBox, QFileDialog, QGridLayout, QLabel, QLineEdit,
  QPushButton,
)

from ..core.paths import (
  default_hf_home, default_lyrics_dir, default_stem_cache, resolve_dir,
)
from ..core import translate
from .util import human_size, settings

# label, QSettings key, env var (None = no env fallback), default factory
ROWS = [
  ("Stem cache", "paths/stem_cache", None, default_stem_cache),
  ("Lyrics dir", "paths/lyrics_dir", "LYRICS_DIR", default_lyrics_dir),
  ("HF models", "paths/hf_home", "HF_HOME", default_hf_home),
]

# plain text prefs for the LLM translate action: label, QSettings key, default
TR_ROWS = [
  ("LLM endpoint", "translate/endpoint", translate.ENDPOINT_DEFAULT),
  ("LLM model", "translate/model", translate.MODEL_DEFAULT),
  ("Translation target", "translate/target", translate.TARGET_DEFAULT),
]

def dir_size(path):
  """Total bytes under `path`, None when it does not exist. Unreadable or
  vanishing entries are skipped rather than aborting the walk."""
  if not path.is_dir():
    return None
  total = 0
  for f in path.rglob("*"):
    try:
      if f.is_file():
        total += f.stat().st_size
    except OSError:
      continue
  return total

class SettingsDialog(QDialog):
  """Three path rows; OK writes the non-empty ones into QSettings."""

  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Settings")
    grid = QGridLayout(self)
    st = settings()
    self._edits = []
    for row, (label, key, env, default) in enumerate(ROWS):
      edit = QLineEdit(st.value(key, "", str) or "")
      edit.setMinimumWidth(360)
      # placeholder shows what an empty field resolves to right now
      edit.setPlaceholderText(str(resolve_dir(key, env, default())))
      browse = QPushButton("Browse...")
      browse.clicked.connect(lambda _=False, e=edit: self._browse(e))
      used = dir_size(resolve_dir(key, env, default()))
      grid.addWidget(QLabel(label), row, 0)
      grid.addWidget(edit, row, 1)
      grid.addWidget(browse, row, 2)
      grid.addWidget(QLabel("—" if used is None else human_size(used)), row, 3)
      self._edits.append((key, edit))
    base = len(ROWS)
    for i, (label, key, default) in enumerate(TR_ROWS):
      edit = QLineEdit(st.value(key, "", str) or "")
      edit.setMinimumWidth(360)
      edit.setPlaceholderText(default or "")
      grid.addWidget(QLabel(label), base + i, 0)
      grid.addWidget(edit, base + i, 1)
      self._edits.append((key, edit))
    tail = base + len(TR_ROWS)
    note = QLabel("Lyrics and HF paths apply on the next launch.")
    note.setEnabled(False)
    grid.addWidget(note, tail, 0, 1, 4)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
    buttons.accepted.connect(self.accept)
    buttons.rejected.connect(self.reject)
    grid.addWidget(buttons, tail + 1, 0, 1, 4)

  def _browse(self, edit):
    start = edit.text().strip() or edit.placeholderText()
    chosen = QFileDialog.getExistingDirectory(self, "Choose folder", start)
    if chosen:
      edit.setText(chosen)

  def accept(self):
    st = settings()
    for key, edit in self._edits:
      st.setValue(key, edit.text().strip())
    super().accept()
