"""Region edits through the window must survive saving and reopening."""
import pytest
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLineEdit, QToolBar, QStyle, QStyleOptionViewItem

from utasub.core import session, romanize
from utasub.core.align import Cue
from utasub.core.regions import Region
from utasub.gui.main_window import MainWindow
from utasub.gui import theme


@pytest.fixture
def window(tmp_path, monkeypatch):
  lang = romanize.get_lang_code()
  locale = romanize.get_default_locale()
  app = QApplication.instance()
  font, palette, stylesheet = app.font(), app.palette(), app.styleSheet()
  theme.apply_theme(app)
  monkeypatch.setattr(MainWindow, 'probe_llm', lambda self: None)
  win = MainWindow()
  win._media_path = str(tmp_path / 'concert.mkv')
  win._timeline._media_path = win._media_path
  win._regions = [Region(2, 20), Region(30, 50)]
  win._region_scored = {0: [], 1: []}
  win._timeline.canvas.duration = 60
  win._timeline.load_cues([Cue(4, 8, 'sample', 'lrc')], regions=win._regions)
  yield win
  win._timeline.canvas.clear_dirty()
  win.close()
  app.setFont(font)
  app.setPalette(palette)
  app.setStyleSheet(stylesheet)
  romanize.set_lang_override(lang or 'auto')
  romanize.set_default_locale(locale)


def test_boundary_drag_enables_save_and_persists(window):
  window._timeline.canvas.move_region_bound(0, 'end', 24)
  assert window._save_act.isEnabled()
  window._on_save()
  assert session.load(window._media_path)['regions'][0]['end'] == 24
  assert not window._save_act.isEnabled()


def test_exact_bounds_reject_overlap_then_save(window):
  def enter_times():
    dialog = QApplication.activeModalWidget()
    start, end = dialog.findChildren(QLineEdit)
    buttons = dialog.findChild(QDialogButtonBox)
    start.setText('0:03.25')
    end.setText('0:35.00')
    buttons.button(QDialogButtonBox.Ok).click()
    assert window._regions[0].end == 20
    assert dialog.isVisible()
    end.setText('0:22.75')
    buttons.button(QDialogButtonBox.Ok).click()

  QTimer.singleShot(0, enter_times)
  window._edit_region_bounds(0)
  assert window._save_act.isEnabled()
  window._on_save()
  region = session.load(window._media_path)['regions'][0]
  assert (region['start'], region['end']) == (3.25, 22.75)


def test_region_save_before_alignment_does_not_save_preview_cues(window):
  window._timeline.show_provisional_cues([Cue(4, 8, 'preview', 'provisional')])
  window._timeline.canvas.move_region_bound(0, 'end', 24)
  window._on_save()
  saved = session.load(window._media_path)
  assert saved['regions'][0]['end'] == 24
  assert saved['cues'] == []
  assert not window._save_act.isEnabled()


def test_exact_bounds_unchanged_preserves_precision(window):
  window._regions[0].start = 2.006
  window._regions[0].end = 20.004
  window._regions[1].start = 20.004
  accepted = []

  def accept_bounds():
    dialog = QApplication.activeModalWidget()
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
    accepted.append(not dialog.isVisible())
    dialog.reject()

  QTimer.singleShot(0, accept_bounds)
  window._edit_region_bounds(0)
  assert accepted == [True]
  assert (window._regions[0].start, window._regions[0].end) == (2.006, 20.004)
  assert not window._save_act.isEnabled()


def test_resize_preserves_fit_and_clamps_zoom_near_media_end(window):
  window._tabs.setCurrentWidget(window._timeline)
  window.show()
  QApplication.processEvents()
  canvas = window._timeline.canvas
  canvas.fit_all()
  for width in (1500, 1100):
    window.resize(width, window.height())
    QApplication.processEvents()
    assert (canvas.view_start, canvas.view_end) == (0, 60)
  canvas.zoom_to_span(50, 60)
  window.resize(1500, window.height())
  QApplication.processEvents()
  assert 0 <= canvas.view_start < canvas.view_end == 60


@pytest.mark.parametrize('offset, expected', [(-8, (0, 'end')), (-1, (0, 'end')),
                                              (1, (1, 'start')), (8, (1, 'start'))])
def test_shared_region_flags_select_boundary_by_side(window, offset, expected):
  canvas = window._timeline.canvas
  canvas.regions = [Region(0, 30), Region(30, 60)]
  canvas.fit_all()
  assert canvas._hit_region(canvas.time_to_x(30) + offset, canvas._ruler_y() + 3) == expected


def test_toolbar_actions_remain_compact_when_enabled(window):
  window.show()
  QApplication.processEvents()
  toolbar = next(bar for bar in window.findChildren(QToolBar) if bar.windowTitle() == 'Session')
  for action, label in [(window._export_act, 'SRT'), (window._export_ass_act, 'ASS'),
                         (window._embed_act, 'Embed'), (window._lrc_as_is_act, 'LRC as-is')]:
    action.setEnabled(True)
    assert toolbar.widgetForAction(action).text() == label
  assert window._transcribe_act not in toolbar.actions()


def test_window_resize_keeps_waveform_height_and_time_scale(window):
  window._tabs.setCurrentWidget(window._timeline)
  window.show()
  QApplication.processEvents()
  canvas = window._timeline.canvas
  canvas.zoom_to_span(5, 15)
  height = canvas.height()
  cue_width = canvas.time_to_x(10) - canvas.time_to_x(5)
  start = canvas.view_start
  window.resize(window.width() + 300, window.height() + 200)
  QApplication.processEvents()
  assert canvas.height() == height
  assert canvas.view_start == start
  assert canvas.time_to_x(10) - canvas.time_to_x(5) == pytest.approx(cue_width)


def test_cue_numbers_fit_after_loading_more_rows(window):
  grid = window._timeline.grid
  window._tabs.setCurrentWidget(window._timeline)
  window.show()
  grid.load_cues([Cue(i, i + 0.5, '', 'lrc') for i in range(1100)])
  QApplication.processEvents()
  item = grid.item(1099, grid.COL_NUM)
  assert item.text() == '1100'
  grid.scrollToItem(item)
  option = QStyleOptionViewItem()
  option.initFrom(grid)
  option.rect = grid.visualItemRect(item)
  grid.itemDelegate().initStyleOption(option, grid.indexFromItem(item))
  text_rect = grid.style().subElementRect(QStyle.SE_ItemViewItemText, option, grid)
  assert option.fontMetrics.elidedText(item.text(), Qt.ElideRight, text_rect.width()) == item.text()
