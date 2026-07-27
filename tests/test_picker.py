"""Unit tests for PickerPanel (offscreen via conftest QApplication)."""
from utasub.core.providers import Candidate
from utasub.gui.picker import PickerPanel, build_alternates


def _fake_scored():
  """Three candidates spanning above/below threshold."""
  return [
    (0.72, Candidate("Good Match", "Album A", "Artist X",
                     "NetEase", 180000, "[00:01.00] line one\n[00:05.00] line two")),
    (0.45, Candidate("OK Match", "Album B", "Artist Y",
                     "LRCLib", 200000, "[00:01.00] other text")),
    (0.20, Candidate("Bad Match", "Album C", "Artist Z",
                     "NetEase", 150000, "[00:01.00] wrong song")),
  ]


def _fake_tags():
  return {"title": "Good Match", "album": "Album A", "artist": "Artist X/Artist W"}


def _panel(scored=None):
  """Build a panel and inject scored rows (skips network)."""
  p = PickerPanel(media_path="/fake/Artist X - Good Match.mkv", tags=_fake_tags())
  if scored is not None:
    p.populate(scored)
  return p


def test_populate_adds_scored_rows_plus_paste_row():
  """Table has N scored rows + 1 paste-row."""
  scored = _fake_scored()
  p = _panel(scored)
  assert p.table.rowCount() == len(scored) + 1


def test_prefill_alternates_from_tags():
  """build_alternates surfaces title + split artists from tags; panel binds
  first title alternate into its combo on construction."""
  alts = build_alternates("/fake/Artist X - Good Match.mkv", _fake_tags())
  assert "Good Match" in [v for v, _ in alts["title"]]
  artists = [v for v, _ in alts["artist"]]
  assert "Artist X" in artists and "Artist W" in artists
  assert _panel().title_combo.lineEdit().text() == "Good Match"


def test_paste_row_enables_editing_and_returns_manual_candidate():
  """Selecting paste-row makes preview editable and yields a Manual candidate."""
  scored = _fake_scored()
  p = _panel(scored)
  p.table.selectRow(len(scored))
  assert not p.preview.isReadOnly()
  p.preview.setPlainText("[00:01.00] manual line")
  ch = p.chosen_candidate()
  assert ch is not None and ch[1].source == "Manual"
  assert "[00:01.00] manual line" in ch[1].lrc


def test_paste_text_persists_across_row_switch():
  """Pasted text survives switching to another row and back."""
  scored = _fake_scored()
  p = _panel(scored)
  paste_row = len(scored)
  p.table.selectRow(paste_row)
  p.preview.setPlainText("[00:02.00] kept line")
  p.table.selectRow(0)
  p.table.selectRow(paste_row)
  assert p.preview.toPlainText() == "[00:02.00] kept line"


def test_selecting_row_returns_that_candidate():
  """chosen_candidate returns the selected row's candidate."""
  scored = _fake_scored()
  p = _panel(scored)
  p.table.selectRow(1)
  ch = p.chosen_candidate()
  assert ch is not None and ch[1].title == "OK Match"


def test_empty_candidates_shows_only_paste_row():
  """Zero candidates: paste-row only, editable."""
  p = _panel([])
  assert p.table.rowCount() == 1
  p.table.selectRow(0)
  assert not p.preview.isReadOnly()


def test_row_change_only_previews_and_apply_button_follows_selection():
  """Row selection previews only; nothing applies until button fires.
  Button tracks whether selected row has usable lyrics."""
  scored = _fake_scored()
  p = _panel()
  fired = []
  p.apply_requested.connect(lambda: fired.append(1))
  p.populate(scored)
  p.table.selectRow(1)
  assert not fired
  assert p.apply_btn.isEnabled()

  p.table.selectRow(len(scored))  # empty paste-row: nothing to apply
  assert not p.apply_btn.isEnabled()
  p.preview.setPlainText("[00:01.00] manual line")
  assert p.apply_btn.isEnabled()
  p.apply_btn.click()
  assert fired


def test_superseded_search_results_are_ignored():
  """Late results from a parked worker must not repopulate table or re-enable
  search; only current worker's signals count."""
  p = _panel(_fake_scored())

  class _Stale:
    pass

  stale = _Stale()
  p._worker = object()  # a different, current worker
  # simulate a late signal from a superseded worker
  orig_sender = type(p).sender
  type(p).sender = lambda self: stale
  try:
    p._on_results([])
    assert p.table.rowCount() == len(_fake_scored()) + 1  # untouched
    p._on_error("boom")
    assert "Error" not in p.status_label.text()
  finally:
    type(p).sender = orig_sender
