"""Benchmark the two Villa revisions with the same Zarr crop (run beside fetch_roi.py)."""
from pathlib import Path
from statistics import median
from time import perf_counter
from types import ModuleType, SimpleNamespace
from dataclasses import make_dataclass
import hashlib
import sys

import numpy as np
import zarr

def load_module(path):
    module = ModuleType("discovery")
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module

if __name__ == "__main__":
    base, patched = [Path(p) for p in sys.argv[1:3]]
    assert hashlib.sha256(base.read_bytes()).hexdigest() == "439b6b0c25c1d496475399651ac3f3cbb03e96d6a76fdb4f2556f5acc4d03896"
    # The function only constructs output records. Stub the record and Segment type.
    for name in ("vesuvius", "vesuvius.ink_detection", "vesuvius.ink_detection.types"):
        sys.modules[name] = ModuleType(name)
    types_module = sys.modules["vesuvius.ink_detection.types"]
    types_module.Patch = make_dataclass("Patch", ["segment", "bbox", ("is_unlabeled",bool,False)])
    types_module.Segment = object
    image = np.load("PHercParis4_20231022170901_level3_z54.npy", allow_pickle=False)
    volume = zarr.open_array("crop.zarr",mode="w",shape=image.shape,chunks=(1,128,128),dtype="u1",zarr_format=2)
    volume[:] = image
    config = SimpleNamespace(scan_scale=None, overlap=0.5)
    segment = SimpleNamespace(image_volume="image", scale=0, patch_size=(3,256,256),
                              supervision_mask=None, validation_mask=None,
                              data_config=SimpleNamespace(patch_finding=config))
    modules = [load_module(base),load_module(patched)]
    timings = [[],[]]
    for rep in range(5):
        outputs = {}
        for i in ((0,1) if rep % 2 == 0 else (1,0)):
            start = perf_counter()
            train, heldout = modules[i].find_segment_unlabeled_patches(segment,lambda name,scale: volume)
            timings[i].append(perf_counter()-start)
            outputs[i] = [p.bbox for p in train]
            assert not heldout
        assert outputs[0] == outputs[1] and len(outputs[0]) == 64
    print("Seconds (before, after):", *map(median,timings))
