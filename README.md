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
