# utasub

<div align="center">

[![Stars](https://img.shields.io/github/stars/nnyj/utasub?style=for-the-badge&labelColor=555&color=e3b341)](https://github.com/nnyj/utasub/stargazers)
[![Build](https://img.shields.io/github/actions/workflow/status/nnyj/utasub/release.yml?style=for-the-badge&labelColor=555)](https://github.com/nnyj/utasub/actions)

</div>

- Stamps official lyrics onto performance videos, outputs `.srt`/`.ass` subtitles
- Fetches lyrics from NetEase/LRCLib, aligns them to a cached ASR transcript with forced alignment and vocal-envelope snapping
- Timeline GUI for manual fixes, machine proposes, human picks and drags

![Timeline: waveform with draggable cues, region list, cue table with romaji](images/timeline.png)

![Picker: per-region lyric candidates with similarity score, credit lines struck out](images/picker.png)

## Features

- One command per file: `utasub concert.mkv` transcribes when the file has no ASR block, then fetches, aligns and opens the GUI
- Batch targets: files, directories (`--recursive`) and globs, own outputs (`.subbed`, `.vocals`, `.utasub`) skipped, one summary table at the end
- Lyric fetch from NetEase (primary), LRCLib, embedded container tags, and a curated local `.lrc` library
- Candidate ranking against the ASR transcript in romaji space (text + duration similarity)
- Credit-line cleanup via `lyrickit.classify_lines`/`filter_credit_lines`: strikes 作词/作曲/编曲 labels, watermarks, LRC headers
- Placement chain: LRC warp (piecewise tempo map) with coarse + forced-alignment fallback
- Qwen3-ForcedAligner word-level stamping with plausibility gates, still runs without a GPU
- Multi-song mode: region detection from vocal gaps, setlist discovery (Wikipedia) or pasted setlist, order-aware DP matching
- Suspect regions: a weak or near-tied auto-assignment is flagged amber in the region list, tooltip names the runner-up title
- MC detection: spoken intermission talk flagged from transcript text signals, excluded from lyric matching, exported as grey romaji cues
- Junk filter: ASR text invented over crowd noise, cheers and tiled chants dropped before export (`--asr-fill`)
- PySide6 GUI: per-region lyrics picker, `Align > Paste setlist...` re-match, waveform timeline with draggable cues, onset snap, tap-sync, undo/redo
- Conf column per cue with the CTC score, amber below this song's cut, `Alt+N`/`Alt+P` jump between the low-confidence lines
- Keyboard cue editing: `N`/`P` step cues, `Left`/`Right` nudge 10 ms, `Shift` moves the end only, `Ctrl` nudges 100 ms, `Ctrl+Space` plays from the cue, `View > Shortcuts` lists the whole map
- Karaoke `\kf` fill on the sung line, timed from the CTC token spans of the adopted stamp
- `--translation` adds a third `.ass` line from the NetEase `tlyric` already in the fetch cache, matched to the LRC line by timestamp, previewed greyed in the picker
- `File > Export LRC` writes the region's corrected timings back to the curated library as `<artist> - <title>.lrc`, rebased to the region start
- `Playback > Preview in mpv` plays the video with the exported `.ass` from the playhead (needs `mpv` on PATH)
- Deterministic re-export from a session file, no network or re-alignment needed
- Output: `.srt` (plus `.ja.srt`/`.romaji.srt`), styled dual-line `.ass`, mux into `.subbed.mkv` with the `.ass` as the default track and the `.srt` as fallback

## Usage

```sh
utasub concert.mkv                # transcribe if needed, fetch, align, open GUI
utasub concert.mkv --multi-song   # concert with multiple songs
utasub concert.mkv --no-gui       # headless best-effort export
utasub concert.mkv --timeline     # reopen saved session, skip align
utasub lives/ --recursive         # every media file under a tree, summary table at the end
```

- `utasub-asr lives/*.mkv` pre-transcribes a batch ahead of time, writing the ASR block into each `<name>.utasub.json`
- `--no-asr` skips in-process transcription and leaves a file without an ASR block untouched
- `utasub.bat` accepts drag-and-drop and runs `--no-gui`

## CLI flags

- Positional targets: media files, directories or globs, omit them to open the GUI empty

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--gui/--no-gui` | auto | GUI when a display is available |
| `--recursive` | off | descend into subdirectories of a directory target |
| `--multi-song` | off | region detection + per-song matching |
| `--asr/--no-asr` | on | transcribe in-process when the file has no ASR block |
| `--asr-fill/--no-asr-fill` | on | keep transcript outside song spans as cues, junk filtered |
| `--providers` | `NetEase` | comma-separated lyric sources |
| `--fa/--no-fa` | auto | forced alignment, on when `torchaudio` or `qwen_asr` importable |
| `--romaji/--no-romaji` | on | romaji subtitle track |
| `--lang` | `auto` | lyric language for romanization: `auto`, `jp`, `cn`, `kr` |
| `--setlist/--no-setlist` | on | setlist discovery in multi-song mode |
| `--setlist-file` | none | manual setlist text file, overrides discovery |
| `--mc/--no-mc` | on | detect spoken MC talk, exclude from matching, subtitle it as grey romaji |
| `--timeline` | off | open saved session directly, skip align |
| `--fresh` | off | bypass fetch cache |
| `--offset` | `-0.2` | SRT offset seconds, negative = earlier |
| `--lead` | `0.0` | extra head start seconds |
| `--ass-font` | `Arial` | `.ass` font name |
| `--ass-size` | `72` | `.ass` font size, PlayRes px |
| `--ass-outline` | `3` | `.ass` outline thickness |
| `--ass-box` | off | opaque box behind the text (BorderStyle 3) |
| `--ass-pos` | `2` | subtitle position: 2 bottom, 8 top |
| `--translation/--no-translation` | off | third `.ass` line from the candidate's translated lyrics |

- `utasub-asr` flags: `--lang` (force language), `--small` (0.6B model), `--cpu`, `--no-stems` (transcribe full mix), `--overwrite`

## How it works

```mermaid
flowchart TD
  A[video/audio file] --> B[ASR: vocal stem + Qwen3-ASR transcript, cached in session JSON]
  B --> C[region detection from vocal gaps, multi-song only]
  C --> D[lyric fetch: NetEase / LRCLib / tags / local library]
  D --> E[candidate scoring vs transcript in romaji space]
  E --> F{placement chain}
  F -->|1| G[lrc_warp: global CTC anchors + piecewise tempo map of LRC timestamps]
  F -->|2, fallback| H[coarse_fa: romaji char map + Qwen FA stamps, envelope-only if sparse]
  G --> I[polish: onset snap, silence rescue, gated CTC stamp adoption]
  H --> I
  I --> J[timeline GUI, manual fixes]
  J --> K[export: srt / ass / mux]
```

- ASR: Qwen3-ASR runs on a Mel-Band Roformer vocal stem and the transcript is cached inside `<name>.utasub.json`, `utasub` transcribes on demand and `utasub-asr` does the same as a batch pre-pass
- Candidate scoring: fetched lyric candidates are scored against the transcript in romaji space, credits stripped
- Placement runs per song as a strategy chain, first success wins: `lrc_warp` then `coarse_fa`
- Two aligners: a global CTC pass (torchaudio MMS_FA) pins the warp for `lrc_warp` and scores, stamps and end-trims both strategies' cues, Qwen3-ForcedAligner gives the per-line stamps `coarse_fa` places from
- Without an aligner (`--no-fa`, no `torchaudio`, no GPU): `lrc_warp` cannot fit (no anchors), `coarse_fa` degrades to envelope-only
- Multi-song: detected regions are matched to the setlist by monotonic DP
- All state (candidates, cues, CTC scores and token spans, manual edits, offsets) lives in the session JSON, so export is deterministic and repeatable
- The session stores the media basename, so moving the pair to another folder keeps it valid; manual edits are kept as `timings`, `texts`, `deleted` and `added` records, so a re-align keeps hand work
- Karaoke: the adopted stamp's per-token spans are stored relative to the cue start and rendered as `\kf` runs on the sung line of the `.ass`, unmatched romaji rides the run before it

## Glossary

- LRC: lyric file format with per-line timestamps (`[mm:ss.xx] line`), here always the studio recording's timings
- ASR: automatic speech recognition, audio in, text with rough segment times out
- FA, forced alignment: the inverse of ASR, the lyric text is already known, the model listens to the audio and measures when each token of it is sung
- FA stamp: aligner output aggregated per lyric line, one measured (start, end) pair plus a confidence
- Coarse map: rough per-line start from text matching alone, no model, used for candidate and region scoring
- Anchor: a (lyric line, audio time) pair the warp is fitted through, here every line the global CTC pass stamped
- CTC, connectionist temporal classification: the decoding scheme forced alignment rests on, the acoustic model scores every token's probability per audio frame, and the best monotonic path through that grid that spells the known text gives each token its time span
- Global CTC pass: one such path over a whole song's audio at once, every line lands in order and on the right repeat of a chorus
- Star token: a wildcard label in the target sequence that any frame may match, placed between lines so instrumental breaks and MC talk do not drag a lyric token onto them
- Confidence gate: drop the lowest-scoring tenth of lines by mean token log-prob, applied to the CTC line ends and to the `coarse_fa` stamps, not to start adoption
- Vocal stem: the separated vocals-only audio track (accompaniment removed)
- Envelope: loudness-over-time curve computed from the vocal stem, tells voiced from silent spans, used for placement checks and end trimming
- Onset: sudden spectral-flux energy rise, a likely syllable/note start, snap target for line starts
- DP, dynamic programming: exhaustive-but-cheap optimization over ordered choices, used twice, changepoint search for warp segment boundaries, and monotonic region-to-setlist matching
- Warp: the fitted map from LRC time to live-audio time, piecewise linear, each piece is a tempo ratio plus offset

## Alignment algorithms

- LRC warp (`core/warp.py`, `core/place.py`):
  - Treats the studio LRC timestamps as a correct shape, models the live take as that shape shifted and stretched
  - Fits a monotone piecewise-linear map from LRC time to audio time, each piece has a tempo slope and offset
  - Piece boundaries are chosen by a changepoint dynamic program, a fixed penalty per break means it only splits when the split buys real accuracy
  - Discontinuous jumps (inserted/cut material) are only allowed where the audio is actually silent
- Anchors (`core/anchors.py`, `core/ctc_align.py`):
  - One global monotonic CTC pass stamps every line: torchaudio's MMS_FA acoustic model over the whole region span, targets are all lines' romaji tokens in order with a `<star>` wildcard between them (and at head/tail) to absorb intros, solos and MC talk
  - The single Viterbi path puts every line in order by construction, so a repeated chorus cannot be matched to the wrong occurrence
  - Confidence per line is its mean token log-prob. Anchors go to the warp fit ungated (the fit's own chain filters discard outliers); the CTC line ends are gated at the per-song 10th percentile
  - The same score drives the GUI Conf column, amber under `low_conf_cut`: the bottom 10% of a song whose median stamp exceeds 0.5 probability, otherwise the absolute 0.2 floor, so a bad align flags every line rather than a tenth of them
  - Emission is computed in 30s chunks with 1.5s of context per seam (self-attention is quadratic in frames), the alignment stays one pass over the concatenated emission
- Forced alignment (`core/fa.py`):
  - Qwen3-ForcedAligner gets known lyric text plus a padded audio window, returns per-token times mapped back to lines
  - Stamps discarded if the implied singing rate is implausible or the stamp drifts too far from its seed
  - Only the `coarse_fa` fallback uses it; the warp's anchors come from the CTC pass
- Coarse map (`core/align.py`):
  - A rough start time per lyric line derived from text matching alone, no FA model involved
  - ASR only times whole segments, so each character inside a segment gets an invented time by linear interpolation between segment start and end
  - Lyrics and transcript are romanized to character streams, `SequenceMatcher` finds matching runs, a lyric line starts at the interpolated time of its first matched character
  - `coarse_fa` strategy (the fallback when no warp fits, so the path a plain-text lyric without timestamps takes): every line starts at its coarse-map time, then any line the FA model stamped (and that survived the gates) has its time replaced by the stamp
  - If FA stamped under half the lines the stamps are distrusted entirely, placement uses coarse-map times nudged onto voiced spans of the envelope (envelope-only)
  - Either way the global CTC pass then runs over the region: every line gets its score for the Conf column, a stamp within the same `ADOPT_MAX` 2.5s bound the warp path uses replaces the start and carries its token spans for the `\kf` fill, and the gated CTC line end trims the cue end
- Placement polish (`core/place.py`):
  - Final starts get local corrections: snap to nearest strong spectral-flux onset (`SNAP_WIN` 0.5s), forward rescue out of silence (`SILENCE_REACH` 2.0s), then adoption of the CTC stamps (the same pass that gave the anchors, no second alignment)
  - Adoption has one filter, the `ADOPT_MAX` 2.5s bound from the warped-and-snapped placement: the warp is fitted from these same stamps, so a stamp further out disagrees with every other stamp in the song (a line smeared across an instrumental)
  - No confidence gate on adoption, it drops correct low-score stamps on quiet or breathy entries that the bound already vets; over 766 gold lines gating costs 14 lines by more than 0.3s and 3 by more than 2s
  - An adopted stamp also carries its per-token spans onto the cue, which is where the `.ass` karaoke fill comes from; a snapped line gets no spans, since they would time audio a snapped cue moved off
  - Ends come from the LRC interval scaled by segment tempo, trimmed by envelope and the CTC line end, bounded by a chars-per-second rate guard

## Tests

- `pytest tests/` runs the synthetic suite, no media or network needed

## Install

```sh
pip install -e .          # core: cutlet, numpy
pip install -e .[gui]     # + PySide6
pip install -e .[asr]     # + qwen-asr, audio-separator (pulls torch)
```

- Requires Python 3.12+, `ffmpeg`/`ffprobe` on PATH
- [lyrickit](https://github.com/nnyj/lyrickit) and [romakit](https://github.com/nnyj/romakit) (credit cleanup, romanization) install automatically from GitHub; for development, a local `pip install -e path/to/lyrickit` afterwards overrides the git copy
- No local services or API keys needed, network use is limited to the lyric providers and the Wikipedia API
- Optional paths: curated lyrics library dir (provider config), stem model dir (`UTASUB_STEM_MODEL_DIR`, defaults to library cache)

## Credits

- [qwen-asr](https://github.com/QwenLM/Qwen3-ASR) : transcription and forced alignment
- [torchaudio MMS_FA](https://pytorch.org/audio/stable/generated/torchaudio.pipelines.MMS_FA.html) : global CTC anchor pass
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) : vocal stem separation

## License

[MIT](LICENSE)
