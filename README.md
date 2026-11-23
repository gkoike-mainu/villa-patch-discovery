# Faster unlabeled patch discovery in Villa

This patch speeds up generation of candidate training regions in the Vesuvius Challenge ink-detection pipeline. The upstream function already loads the middle image plane. Reusing this plane avoids a second series of Zarr reads. A boolean grid marks occupied stride cells and emits them in the same row-major order as the original sort. Mask processing and the minimum coverage criterion are unchanged.

## Measurement

On a 1,024 × 1,024 crop of PHerc. Paris 4, the reference function took a median of 0.480 seconds and the patched function 0.025 seconds over five alternating runs (18.9×). Both returned exactly the same 64 ordered bounding boxes. A second run using the standalone reproduction script measured 0.477 and 0.030 seconds respectively. These numbers describe this preprocessing function on a dense local crop. We did not measure whole-volume memory, training throughput, ink accuracy or newly recovered text. No supervision or validation mask was supplied in this test.

## Data and reproduction

The input comes from the official Vesuvius public S3 archive, segment 20231022170901, surface volume 2.4um-0.22m-78keV-volume-20260411134726.zarr, level 3, plane 54, rows 768–1791 and columns 6656–7679 (inclusive). The fetch script downloads only the relevant byte ranges. It stores the crop locally; this repository contains no scan pixels. The local benchmark uses 1 × 128 × 128 chunks, patch size 256 and stride 128. The benchmark checks the upstream file hash and compares every box in order on each run. Python 3.12, NumPy, Zarr 3.4.2, curl and patch were used.

Run these commands in a fresh directory containing the release files:

```sh
python -m pip install numpy zarr==3.4.2
python fetch_roi.py
curl -fsSLo upstream.py https://raw.githubusercontent.com/ScrollPrize/villa/c8f3ba45304f0a317a49a155929e30d294f201eb/vesuvius/src/vesuvius/ink_detection/data/patch_finding_default.py
cp upstream.py patched.py
patch patched.py < villa_unlabeled.patch
python reproduce.py upstream.py patched.py
```

To apply the change to the pinned Villa checkout, run `git apply /path/to/villa_unlabeled.patch` from its root. The permissive license covers the contributed changes and benchmark scripts. Upstream code and public scan data retain their original terms.

## Whole-plane replication, 23 November 2026

The complete 669 × 3,620 middle plane at pyramid level 5 includes irregular edges and zero-filled areas. Both revisions returned the same 149 training boxes in the same order and zero held-out boxes. Four alternating runs per revision in fresh processes had median discovery times of 1.865 s upstream and 0.109 s patched (17.17×). Timings exclude network transfer and local Zarr construction. This measures patch discovery on the middle plane, not end-to-end model training.

After applying the patch as above, run `python whole_plane.py upstream.py patched.py`. The script fetches the public scan, validates its metadata and compares ordered output hashes. [Raw measurements](measurement_level5.json).

## Second segment and three-plane patch reads

The revised patch uses NumPy reductions over rows and columns to find occupied stride cells. This avoids creating coordinate arrays for every nonzero pixel. The minimum coverage rule, clipping and traversal order still apply. Use `villa_unlabeled_v2.patch` directly on the pinned upstream revision. The original patch remains available to reproduce the earlier measurements.

The new benchmark uses depth planes 53–55 of segment 20231005123336, level 5. It stores the 3 × 1,090 × 3,040 slab as local Zarr chunks. For every region identified on the middle plane, the benchmark reads the surrounding three-layer patch and records its bounds, shape, flag, intensity sum and nonzero count. All three implementations yielded identical ordered records for 205 regions. Four fresh processes per implementation ran in alternating order.

| Median | Upstream | First patch | Revised patch |
| --- | ---: | ---: | ---: |
| Discover candidate regions (seconds) | 2.480 | 0.138 | 0.081 |
| Discover and read all patches (seconds) | 2.818 | 0.686 | 0.432 |
| Peak process RSS (MiB) | 192.4 | 136.8 | 74.1 |

These measurements cover one local, dense slab without supervision masks and exclude initial data download. The complete benchmark includes imports in peak memory. Slab coordinates use local depth indices 0–2. A separate deterministic test covers masked inputs, nearly empty arrays, negative values, NaNs, different scan scales, clipped windows and different strides in 320 cases. End-to-end training, segmentation accuracy and text recovery were outside the benchmark.

To reproduce, fetch the same upstream file using the command above, then run the commands below on Linux with NumPy and Zarr installed.

```sh
patch -o previous.py upstream.py < villa_unlabeled.patch
patch -o candidate.py upstream.py < villa_unlabeled_v2.patch
python test_equivalence.py upstream.py candidate.py
python benchmark_slab.py ./scan-cache upstream.py previous.py candidate.py
```

[Measurements with per-run timings and fingerprints](measurement_second_segment.json).
