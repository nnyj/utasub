"""Warp fit diagnostics: dropped anchors must count every filter, not just the
head guard, so a repeat-tail drop is visible in the log."""

def test_diag_dropped_counts_chain_filter_drops():
  from utasub.core.warp import fit_warp
  lrc = [i * 5.0 for i in range(24)]
  # 20 clean anchors, then 4 tail stamps landing on an earlier repeat
  anchors = [(j, lrc[j] + 2.0) for j in range(20)]
  anchors += [(j, lrc[j] - 30.0) for j in range(20, 24)]
  segs, diag = fit_warp(anchors, lrc, [(2.0, 130.0)])
  assert diag["anchors"] == 20, diag
  assert diag["dropped"] == 4, diag

def test_diag_dropped_is_zero_on_clean_anchors():
  from utasub.core.warp import fit_warp
  lrc = [i * 5.0 for i in range(20)]
  anchors = [(j, lrc[j] + 2.0) for j in range(20)]
  segs, diag = fit_warp(anchors, lrc, [(2.0, 110.0)])
  assert diag["dropped"] == 0, diag
  assert diag["anchors"] == 20, diag
