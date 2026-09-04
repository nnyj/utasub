"""QAction / menu / toolbar construction for MainWindow.
Actions the window later enables or toggles are stored on it as attributes."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QToolBar, QMessageBox

from . import theme
from .util import settings

# Panel-scoped keys: they fire while focus is in the timeline tab, so the grid
# keeps Up/Down for row navigation and the rest of the window keeps its arrows.
TIMELINE_KEYS = [
  ("N", "step_cue", (1,), {}),
  ("P", "step_cue", (-1,), {}),
  ("Left", "nudge_cue", (-1,), {}),
  ("Right", "nudge_cue", (1,), {}),
  ("Shift+Left", "nudge_cue", (-1,), {"edge": "end"}),
  ("Shift+Right", "nudge_cue", (1,), {"edge": "end"}),
  ("Ctrl+Left", "nudge_cue", (-1,), {"coarse": True}),
  ("Ctrl+Right", "nudge_cue", (1,), {"coarse": True}),
]

SHORTCUTS = [
  ("Space", "Play / pause"),
  ("Ctrl+Space", "Play from the selected cue, 1 s lead-in"),
  ("T", "Tap sync: stamp the selected cue's start at the playhead"),
  ("N / P", "Next / previous cue"),
  ("Alt+N / Alt+P", "Next / previous low-confidence cue"),
  ("Left / Right", "Nudge the selected cue by 10 ms"),
  ("Shift+Left / Right", "Nudge the cue's end only"),
  ("Ctrl+Left / Right", "Nudge by 100 ms"),
  ("Ins / Del", "Add / delete cue"),
  ("Ctrl+K / Ctrl+M", "Split at playhead / merge with the next cue"),
  ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
  ("Ctrl+O / Ctrl+S", "Open file / save session"),
  ("Ctrl+Enter", "Picker: apply the selected lyrics to the region"),
  ("Wheel / Shift+Wheel", "Zoom / pan the waveform"),
  ("Drag / Shift+drag / Alt+drag", "Move cue / ripple the following cues / no snap"),
]

def _shortcut_sheet(win):
  """View > Shortcuts: the whole key map in one readable table."""
  rows = "".join(
    f"<tr><td style='padding:2px 16px 2px 0'><b>{k}</b></td><td>{d}</td></tr>"
    for k, d in SHORTCUTS)
  box = QMessageBox(win)
  box.setWindowTitle("Shortcuts")
  box.setTextFormat(Qt.RichText)
  box.setText(f"<table>{rows}</table>")
  box.exec()

def _annotate_shortcuts(win):
  """Append every action's key to its tooltip, so the icon-only toolbar teaches
  the shortcut instead of hiding it in a menu."""
  for act in win.findChildren(QAction):
    key = act.shortcut().toString()
    tip = act.toolTip() or act.text()
    if key and tip and key not in tip:
      act.setToolTip(f"{tip}  ({key})")

def build_actions(win):
  """Create every action, the menu bar and the icon toolbar for `win`."""
  timeline = win._timeline

  open_act = QAction(theme.icon("open"), "Open...", win)
  open_act.setShortcut(QKeySequence("Ctrl+O"))
  open_act.setToolTip("Open media file")
  open_act.triggered.connect(win._on_open)

  save_act = QAction(theme.icon("save", theme.ICON_WRITE), "Save", win)
  save_act.setShortcut(QKeySequence("Ctrl+S"))
  save_act.setToolTip("Save session")
  save_act.triggered.connect(win._on_save)

  win._export_act = QAction(theme.icon("export", theme.ICON_WRITE), "Export SRT", win)
  win._export_act.setEnabled(False)
  win._export_act.setToolTip("Save session + write SRT from the current cues")
  win._export_act.triggered.connect(win._on_export)

  win._export_ass_act = QAction(theme.icon("export_ass", theme.ICON_WRITE), "Export ASS", win)
  win._export_ass_act.setEnabled(False)
  win._export_ass_act.setToolTip("Write styled .ass (romaji over dimmed original) from the current cues")
  win._export_ass_act.triggered.connect(win._on_export_ass)

  win._export_lrc_act = QAction("Export LRC", win)
  win._export_lrc_act.setEnabled(False)
  win._export_lrc_act.setToolTip("Write the region's corrected timings as an .lrc "
                                 "into the curated lyrics library")
  win._export_lrc_act.triggered.connect(win._on_export_lrc)

  exit_act = QAction("Exit", win)
  exit_act.triggered.connect(win.close)

  undo_act = QAction(theme.icon("undo"), "Undo", win)
  undo_act.setShortcut(QKeySequence("Ctrl+Z"))
  undo_act.triggered.connect(timeline.do_undo)

  redo_act = QAction(theme.icon("redo"), "Redo", win)
  redo_act.setShortcut(QKeySequence("Ctrl+Y"))
  redo_act.triggered.connect(timeline.do_redo)

  add_act = QAction(theme.icon("add"), "Add cue", win)
  add_act.setShortcut(QKeySequence(Qt.Key_Insert))
  add_act.triggered.connect(timeline.do_insert)

  del_act = QAction(theme.icon("delete", theme.ICON_DESTRUCTIVE), "Delete cue", win)
  del_act.setShortcut(QKeySequence(Qt.Key_Delete))
  del_act.triggered.connect(timeline.do_delete)

  split_act = QAction(theme.icon("split"), "Split", win)
  split_act.setShortcut(QKeySequence("Ctrl+K"))
  split_act.setToolTip("Split cue at playhead")
  split_act.triggered.connect(timeline.do_split)

  merge_act = QAction(theme.icon("merge"), "Merge", win)
  merge_act.setShortcut(QKeySequence("Ctrl+M"))
  merge_act.setToolTip("Merge cue with the next one")
  merge_act.triggered.connect(timeline.do_merge)

  win._align_region_act = QAction(theme.icon("align_region", theme.ICON_ALIGN), "Align region", win)
  win._align_region_act.setToolTip("Align the selected region into the timeline")
  win._align_region_act.setEnabled(False)
  win._align_region_act.triggered.connect(win._on_align_region)

  win._align_all_act = QAction(theme.icon("align_all", theme.ICON_ALIGN), "Align all", win)
  win._align_all_act.setToolTip("Align all regions into the timeline for editing")
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

  win._embed_act = QAction(theme.icon("embed", theme.ICON_WRITE), "Embed subtitles", win)
  win._embed_act.setToolTip("Mux the exported ASS + SRT into the video (ffmpeg)")
  win._embed_act.setEnabled(False)
  win._embed_act.triggered.connect(win._on_embed)

  win._preview_act = QAction("Preview in mpv", win)
  win._preview_act.setToolTip("Play the video in mpv with the exported .ass, "
                              "from 2 s before the playhead")
  win._preview_act.setEnabled(False)
  win._preview_act.triggered.connect(win._on_preview_mpv)

  # region ops, mirrored from the region dock's context menu
  region_acts = []
  for label, op in (("Add region at playhead", "add"),
                    ("Split region at playhead", "split"),
                    ("Delete region", "delete")):
    act = QAction(label, win)
    act.triggered.connect(lambda _=False, o=op: win.region_op(o))
    region_acts.append(act)

  # transport shortcuts, listed so Space / T are discoverable
  play_act = QAction("Play / Pause", win)
  play_act.setShortcut(QKeySequence(Qt.Key_Space))
  play_act.triggered.connect(lambda: win._play_at())  # drop QAction's checked arg

  play_cue_act = QAction("Play from selected cue", win)
  play_cue_act.setShortcut(QKeySequence("Ctrl+Space"))
  play_cue_act.setToolTip("Seek a second before the selected cue and play")
  play_cue_act.triggered.connect(win._play_from_cue)

  tap_act = QAction("Tap sync", win)
  tap_act.setShortcut(QKeySequence(Qt.Key_T))
  tap_act.setToolTip("Stamp the selected cue's start at the playhead, then advance")
  tap_act.triggered.connect(timeline.tap_sync)

  next_susp_act = QAction("Next low-confidence cue", win)
  next_susp_act.setShortcut(QKeySequence("Alt+N"))
  next_susp_act.setToolTip("Jump to the next cue the aligner was least sure of")
  next_susp_act.triggered.connect(lambda: timeline.step_low_conf(1))

  prev_susp_act = QAction("Previous low-confidence cue", win)
  prev_susp_act.setShortcut(QKeySequence("Alt+P"))
  prev_susp_act.setToolTip("Jump to the previous cue the aligner was least sure of")
  prev_susp_act.triggered.connect(lambda: timeline.step_low_conf(-1))

  shortcuts_act = QAction("Shortcuts...", win)
  shortcuts_act.triggered.connect(lambda: _shortcut_sheet(win))

  cue_key_acts = []
  for keys, method, args, kwargs in TIMELINE_KEYS:
    act = QAction(timeline)
    act.setShortcut(QKeySequence(keys))
    act.setShortcutContext(Qt.WidgetWithChildrenShortcut)
    act.triggered.connect(
      lambda _=False, m=method, a=args, k=kwargs: getattr(timeline, m)(*a, **k))
    timeline.addAction(act)
    cue_key_acts.append(act)
  # a panel shortcut outranks the grid's cell editor, so N/P and the caret keys
  # would never reach the text being typed: stand them down while it is open
  win._cue_key_acts = cue_key_acts
  timeline.grid.editing_changed.connect(
    lambda editing: [a.setEnabled(not editing) for a in cue_key_acts])

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
  file_menu.addAction(win._export_lrc_act)
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
  for act in region_acts:
    region_menu.addAction(act)
  align_menu.addSeparator()
  align_menu.addAction(win._paste_setlist_act)
  align_menu.addAction(win._reset_edits_act)

  play_menu = mb.addMenu("Playback")
  play_menu.addAction(play_act)
  play_menu.addAction(play_cue_act)
  play_menu.addAction(tap_act)
  play_menu.addSeparator()
  play_menu.addAction(win._preview_act)
  play_menu.addAction(next_susp_act)
  play_menu.addAction(prev_susp_act)

  win._view_menu = mb.addMenu("View")
  win._view_menu.addAction(win._region_dock.toggleViewAction())
  win._view_menu.addAction(win._log_dock.toggleViewAction())
  win._view_menu.addSeparator()
  win._view_menu.addAction(shortcuts_act)

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

  _annotate_shortcuts(win)

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
