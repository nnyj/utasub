"""Region detection, query extraction, serialization. No GPU."""

def _make_segments(regions_spec):
  """Create synthetic ASR segments from [(start, end, n_segs, text_template), ...]."""
  segs = []
  for start, end, n, tmpl in regions_spec:
    dur = (end - start) / n
    for i in range(n):
      s = start + i * dur
      segs.append((s, s + dur * 0.8, tmpl.format(i=i)))
  segs.sort(key=lambda x: x[0])
  return segs

def _fake_audio_with_gaps(duration_s, singing_regions, sr=16000):
  """Loud sine in singing regions, silence elsewhere."""
  import numpy as np
  audio = np.zeros(int(duration_s * sr), dtype=np.float32)
  for start, end in singing_regions:
    s_idx = int(start * sr)
    e_idx = min(int(end * sr), len(audio))
    t = np.arange(e_idx - s_idx) / sr
    audio[s_idx:e_idx] = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
  return audio

def test_detect_regions_from_audio_gaps():
  """3 regions, boundaries approx correct."""
  from utasub.core.regions import detect_regions
  audio = _fake_audio_with_gaps(300, [(10, 60), (90, 150), (180, 240)])
  segs = _make_segments([
    (10, 60, 10, "song1 line {i}"),
    (90, 150, 10, "song2 line {i}"),
    (180, 240, 10, "song3 line {i}"),
  ])
  regions = detect_regions(segs, audio, gap_threshold=25, min_region=20)
  assert len(regions) == 3, f"expected 3 regions, got {len(regions)}"
  assert abs(regions[0].start - 10) < 2 and abs(regions[0].end - 60) < 2
  assert abs(regions[1].start - 90) < 2
  assert abs(regions[2].start - 180) < 2

def test_detect_regions_falls_back_to_asr_gaps_without_audio():
  from utasub.core.regions import detect_regions
  segs = _make_segments([
    (10, 60, 10, "song1 line {i}"),
    (100, 160, 10, "song2 line {i}"),
  ])
  regions = detect_regions(segs, None, gap_threshold=25, min_region=20)
  assert len(regions) == 2, f"expected 2 regions, got {len(regions)}"

def test_detect_regions_filters_short_clusters():
  """Cluster shorter than min_region (5s blip) dropped."""
  from utasub.core.regions import detect_regions
  audio = _fake_audio_with_gaps(200, [(10, 60), (90, 95), (130, 180)])
  segs = _make_segments([
    (10, 60, 10, "song {i}"),
    (90, 95, 2, "blip {i}"),
    (130, 180, 10, "song {i}"),
  ])
  regions = detect_regions(segs, audio, gap_threshold=25, min_region=20)
  assert len(regions) == 2, f"expected 2 (5s blip filtered), got {len(regions)}"

def test_region_query_extracts_phrases_for_japanese_and_english():
  """Short JP phrases, frequent EN content words."""
  from utasub.core.regions import region_query, Region

  ja = region_query([
    (10, 15, "変わらない風景朝焼けを硬化した"),
    (20, 25, "白紙の人生に白紙の願い"),
    (30, 35, "変わらないように君が主役"),
    (50, 55, "変わらない人生はどうにも"),
  ], Region(0, 300))
  assert ja and len(ja) < 40
  assert any(ord(c) > 0x3000 for c in ja), f"expected Japanese chars: {ja}"

  en = region_query([
    (10, 15, "Walking down the road again"),
    (20, 25, "Walking through the rain"),
    (30, 35, "She said hello to me"),
    (40, 45, "Walking in the moonlight"),
  ], Region(0, 100))
  assert "walking" in en.lower(), f"expected 'walking' in query: {en}"

def test_region_segments_returns_only_in_bounds():
  from utasub.core.regions import region_segments, Region
  segs = [(10, 15, "a"), (50, 55, "b"), (70, 75, "c"), (110, 115, "d")]
  result = region_segments(segs, Region(50, 100))
  assert [t for _, _, t in result] == ["b", "c"]

def test_region_dict_roundtrip():
  from utasub.core.regions import Region
  r2 = Region.from_dict(Region(10.123, 60.456, candidate_idx=3).to_dict())
  assert abs(r2.start - 10.123) < 0.01
  assert abs(r2.end - 60.456) < 0.01
  assert r2.candidate_idx == 3
