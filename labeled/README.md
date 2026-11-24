# Faster discovery for annotated ink training patches

This patch changes how Villa finds occupied cells on its sampling grid. It reduces the masks in row bands and finds each patch's ink bounding rectangle using its nonempty rows and columns. It preserves the sampling grid, supervision and validation rules, patch coverage and ordering. It applies to labelled and unlabelled surface volumes.

## Full-resolution real-mask benchmark

The official PHerc. Paris. 4 segment `w00_20231016151002` supplies the ink and supervision masks. At 32,249 × 51,380 pixels, both versions produced the same 968 ordered training patches. The benchmark used 512 × 512 patches at 50% overlap and the same 6% minimum label coverage for both versions.

| Median of three fresh processes | Villa upstream | Updated |
|---|---:|---:|
| Discovery time | 8.778 s | 0.862 s |
| Peak process RSS | 4,761 MiB | 2,568 MiB |

The measurement begins after the memory-mapped masks are opened. Time includes access to the mapped image data when a page is first touched, and excludes download, conversion, training and model inference. This is one labelled segment. The benchmark compares patch outputs directly and 480 additional comparisons exercise labelled and unlabelled inputs with and without validation masks.

## Reproduce

Install `numpy`, `tifffile`, `imagecodecs` and the Hugging Face CLI. Copy the two original TIFFs to a local directory, using their short names below.

```sh
mkdir -p masks
source=hf://buckets/scrollprize/datasets/ink/phercparis4/w00_20231016151002/w00_20231016151002
hf buckets cp "${source}_inklabels.tif" masks/inklabels.tif
hf buckets cp "${source}_supervision_mask.tif" masks/supervision.tif
curl -fsSLo upstream.py https://raw.githubusercontent.com/ScrollPrize/villa/c8f3ba45304f0a317a49a155929e30d294f201eb/vesuvius/src/vesuvius/ink_detection/data/patch_finding_default.py
patch -o candidate.py upstream.py < labeled/villa_patch_discovery_v3.patch
python labeled/test_equivalence.py upstream.py candidate.py
python labeled/benchmark.py convert masks 1
python labeled/benchmark.py upstream.py masks 512
python labeled/benchmark.py candidate.py masks 512
```

The benchmark emits the ordered patch list's SHA-256 fingerprint. The full-resolution fingerprint for both versions was `e48e776fb4bba73c6dc54362bfa74961c42304be083f6b139ccaa270013a88c8`. The converter writes two memory-mapped files totaling about 3.1 GiB and does not modify the original TIFFs. This patch follows the repository's MIT license.
