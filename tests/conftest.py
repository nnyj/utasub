"""Shared pytest setup: offscreen Qt, project root on path, one QApplication."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture(scope="session", autouse=True)
def _qapp():
  """Single offscreen QApplication for the whole session (Qt needs one before
  any QWidget is built)."""
  from PySide6.QtWidgets import QApplication
  app = QApplication.instance() or QApplication([])
  yield app
