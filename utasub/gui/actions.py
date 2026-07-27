"""QAction / menu / toolbar construction for MainWindow.
Actions the window later enables or toggles are stored on it as attributes."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QToolBar

from . import theme
from .util import settings


def build_actions(win):
  """Create every action, the menu bar and the icon toolbar for `win`."""
  timeline = win._timeline

  open_act = QAction(theme.icon("open"), "Open...", win)
  open_act.setShortcut(QKeySequence("Ctrl+O"))
  open_act.setToolTip("Open media file (Ctrl+O)")
  open_act.triggered.connect(win._on_open)

  save_act = QAction(theme.icon("save", theme.ICON_WRITE), "Save", win)
  save_act.setShortcut(QKeySequence("Ctrl+S"))
  save_act.setToolTip("Save session (Ctrl+S)")
  save_act.triggered.connect(win._on_save)

  win._export_act = QAction(theme.icon("export", theme.ICON_WRITE), "Export SRT", win)
  win._export_act.setEnabled(False)
  win._export_act.setToolTip("Save session + write SRT from the current cues")
  win._export_act.triggered.connect(win._on_export)

  win._export_ass_act = QAction(theme.icon("export_ass", theme.ICON_WRITE), "Export ASS", win)
  win._export_ass_act.setEnabled(False)
  win._export_ass_act.setToolTip("Write styled .ass (romaji over dimmed original) from the current cues")
  win._export_ass_act.triggered.connect(win._on_export_ass)

  exit_act = QAction("Exit", win)
  exit_act.triggered.connect(win.close)

  undo_act = QAction(theme.icon("undo"), "Undo", win)
  undo_act.setShortcut(QKeySequence("Ctrl+Z"))
  undo_act.setToolTip("Undo (Ctrl+Z)")
  undo_act.triggered.connect(timeline.do_undo)

  redo_act = QAction(theme.icon("redo"), "Redo", win)
  redo_act.setShortcut(QKeySequence("Ctrl+Y"))
  redo_act.setToolTip("Redo (Ctrl+Y)")
  redo_act.triggered.connect(timeline.do_redo)

  add_act = QAction(theme.icon("add"), "Add cue", win)
  add_act.setShortcut(QKeySequence(Qt.Key_Insert))
  add_act.setToolTip("Add cue (Ins)")
  add_act.triggered.connect(timeline.do_insert)

  del_act = QAction(theme.icon("delete", theme.ICON_DESTRUCTIVE), "Delete cue", win)
  del_act.setShortcut(QKeySequence(Qt.Key_Delete))
  del_act.setToolTip("Delete cue (Del)")
  del_act.triggered.connect(timeline.do_delete)

  split_act = QAction(theme.icon("split"), "Split", win)
  split_act.setShortcut(QKeySequence("Ctrl+K"))
  split_act.setToolTip("Split cue at playhead (Ctrl+K)")
  split_act.triggered.connect(timeline.do_split)

  merge_act = QAction(theme.icon("merge"), "Merge", win)
  merge_act.setShortcut(QKeySequence("Ctrl+M"))
  merge_act.setToolTip("Merge cue with next (Ctrl+M)")
  merge_act.triggered.connect(timeline.do_merge)

  win._align_region_act = QAction(theme.icon("align_region", theme.ICON_ALIGN), "Align region", win)
  win._align_region_act.setToolTip("Align the selected region into the timeline")
  win._align_region_act.setEnabled(False)
  win._align_region_act.triggered.connect(win._on_align_region)

  win._align_all_act = QAction(theme.icon("align_all", theme.ICON_ALIGN), "Align all", win)
  win._align_all_act.setToolTip("Align all regions into the timeline for review")
  win._align_all_act.triggered.connect(win._on_align_all)
  win._align_all_act.setEnabled(False)

  win._lrc_as_is_act = QAction(theme.icon("lrc_as_is", theme.ICON_LRC), "Use LRC as-is", win)
  win._lrc_as_is_act.setToolTip("Load the LRC timestamps into the timeline unaligned, "
                                "then nudge them with the Cue offset buttons")
  win._lrc_as_is_act.setEnabled(False)
  win._lrc_as_is_act.triggered.connect(win._on_use_lrc_as_is)

  win._paste_setlist_act = QAction("Paste setlist...", win)
  win._paste_setlist_act.setToolTip("Type or paste a setlist, then re-match the "
                                    "regions against it (manual picks kept)")
  win._paste_setlist_act.triggered.connect(win._on_paste_setlist)

  win._reset_edits_act = QAction("Reset manual edits", win)
  win._reset_edits_act.setToolTip("Forget all hand timings and deletions so the "
                                  "next Align places every line fresh")
  win._reset_edits_act.triggered.connect(win._on_reset_edits)

  win._transcribe_act = QAction(theme.icon("transcribe", theme.ICON_ASR), "Transcribe", win)
  win._transcribe_act.setToolTip("Transcribe via Qwen3-ASR (needs the [asr] extra)")
  win._transcribe_act.triggered.connect(win._on_transcribe)

  win._embed_act = QAction(theme.icon("embed", theme.ICON_WRITE), "Embed SRT", win)
  win._embed_act.setToolTip("Mux the exported SRT into the video (ffmpeg)")
  win._embed_act.setEnabled(False)
  win._embed_act.triggered.connect(win._on_embed)

  # region ops, mirrored from the region dock's context menu
  add_region_act = QAction("Add region at playhead", win)
  add_region_act.triggered.connect(win._on_add_region_act)
  split_region_act = QAction("Split region at playhead", win)
  split_region_act.triggered.connect(win._on_split_region_act)
  del_region_act = QAction("Delete region", win)
  del_region_act.triggered.connect(win._on_delete_region_act)

  # transport shortcuts, listed so Space / T are discoverable
  play_act = QAction("Play / Pause", win)
  play_act.setShortcut(QKeySequence(Qt.Key_Space))
  play_act.triggered.connect(win._toggle_play)

  tap_act = QAction("Tap sync", win)
  tap_act.setShortcut(QKeySequence(Qt.Key_T))
  tap_act.setToolTip("Stamp the selected cue's start at the playhead, then advance")
  tap_act.triggered.connect(timeline.tap_sync)

  cleanup_act = QAction(theme.icon("cleanup", theme.ICON_DESTRUCTIVE), "Cleanup files", win)
  cleanup_act.setToolTip("Delete stray stem / asr / txt files")
  cleanup_act.triggered.connect(win._on_cleanup)
  win._cleanup_act = cleanup_act

  # --- menu bar ---
  mb = win.menuBar()
  file_menu = mb.addMenu("File")
  file_menu.addAction(open_act)
  file_menu.addAction(win._transcribe_act)
  file_menu.addAction(save_act)
  file_menu.addAction(win._export_act)
  file_menu.addAction(win._export_ass_act)
  file_menu.addAction(win._embed_act)
  file_menu.addSeparator()
  file_menu.addAction(cleanup_act)
  file_menu.addSeparator()
  file_menu.addAction(exit_act)

  edit_menu = mb.addMenu("Edit")
  edit_menu.addAction(undo_act)
  edit_menu.addAction(redo_act)
  edit_menu.addSeparator()
  edit_menu.addAction(add_act)
  edit_menu.addAction(del_act)
  edit_menu.addAction(split_act)
  edit_menu.addAction(merge_act)

  align_menu = mb.addMenu("Align")
  align_menu.addAction(win._align_region_act)
  align_menu.addAction(win._align_all_act)
  align_menu.addAction(win._lrc_as_is_act)
  align_menu.addSeparator()
  region_menu = align_menu.addMenu("Region")
  region_menu.addAction(add_region_act)
  region_menu.addAction(split_region_act)
  region_menu.addAction(del_region_act)
  align_menu.addSeparator()
  align_menu.addAction(win._paste_setlist_act)
  align_menu.addAction(win._reset_edits_act)

  play_menu = mb.addMenu("Playback")
  play_menu.addAction(play_act)
  play_menu.addAction(tap_act)

  win._view_menu = mb.addMenu("View")
  win._view_menu.addAction(win._region_dock.toggleViewAction())
  win._view_menu.addAction(win._log_dock.toggleViewAction())

  build_lang_menu(win, mb)

  # --- icon toolbar ---
  tb = QToolBar("Actions")
  tb.setToolButtonStyle(Qt.ToolButtonIconOnly)
  win.addToolBar(tb)
  win._view_menu.addAction(tb.toggleViewAction())
  tb.addAction(open_act)
  tb.addAction(win._transcribe_act)
  tb.addAction(save_act)
  tb.addAction(win._export_act)
  tb.addAction(win._export_ass_act)
  tb.addAction(win._embed_act)
  tb.addSeparator()
  tb.addAction(undo_act)
  tb.addAction(redo_act)
  tb.addSeparator()
  tb.addAction(win._align_region_act)
  tb.addAction(win._align_all_act)
  tb.addAction(win._lrc_as_is_act)
  tb.addSeparator()
  tb.addAction(add_act)
  tb.addAction(del_act)
  tb.addAction(split_act)
  tb.addAction(merge_act)
  tb.addSeparator()
  tb.addAction(cleanup_act)


def build_lang_menu(win, mb):
  """Language menu (inline with File/Edit/...): pins the romanizer,
  'auto' = per-line detect."""
  from ..core.romanize import LANGS, get_lang_code, set_lang_override
  menu = mb.addMenu("Language")
  group = QActionGroup(win)  # parented, exclusive check state
  group.setExclusive(True)
  # a --lang on the command line wins over the stored pref
  saved = get_lang_code() or settings().value("lyrics/lang", "auto", str)
  if saved not in LANGS:
    saved = "auto"
  for lang in LANGS:
    act = QAction("auto-detect" if lang == "auto" else lang, win, checkable=True)
    act.setChecked(lang == saved)
    act.triggered.connect(lambda _c, l=lang: win._on_lang_changed(l))
    group.addAction(act)
    menu.addAction(act)
  set_lang_override(saved)
