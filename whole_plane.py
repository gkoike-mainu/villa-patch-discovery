"""Reproduce discovery on the full middle plane of the public segment at level 5."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter
from types import ModuleType, SimpleNamespace
from dataclasses import make_dataclass

import numpy as np
import zarr

import fetch_roi
from reproduce import load_module


def prepare(work):
    from urllib.request import urlopen
    fetch_roi.ROOT = fetch_roi.ROOT.rsplit("/", 1)[0] + "/5"
    with urlopen(fetch_roi.ROOT + "/.zarray", timeout=30) as response:
        metadata = json.load(response)
    assert (metadata["shape"], metadata["chunks"], metadata["compressor"], metadata["dtype"]) == (
        [109, 669, 3620], [109, 128, 128], None, "|u1")
    assert metadata["fill_value"] == 0 and metadata["order"] == "C"
    h, w = metadata["shape"][1:]
    image = np.empty((h, w), dtype="u1")
    cells = [(r, c) for r in range((h+127)//128) for c in range((w+127)//128)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for r, c, data in pool.map(fetch_roi.fetch, cells):
            block = image[r*128:(r+1)*128, c*128:(c+1)*128]
            block[:] = data[:block.shape[0], :block.shape[1]]
    volume = zarr.open_array(work/"plane.zarr", mode="w", shape=(1,h,w), chunks=(1,128,128), dtype="u1",zarr_format=2)
    volume[0] = image
    return dict(source=fetch_roi.ROOT, middle_plane=54, shape=list(image.shape),
                nonzero=int(np.count_nonzero(image)), sha256=hashlib.sha256(image.tobytes()).hexdigest())


def measure(source, work):
    for name in ("vesuvius","vesuvius.ink_detection","vesuvius.ink_detection.types"):
        sys.modules[name] = ModuleType(name)
    types = sys.modules["vesuvius.ink_detection.types"]
    types.Patch = make_dataclass("Patch", ["segment","bbox",("is_unlabeled",bool,False)])
    types.Segment = object
    segment = SimpleNamespace(image_volume="image", scale=0, patch_size=(3,256,256), supervision_mask=None,
                              validation_mask=None, data_config=SimpleNamespace(patch_finding=SimpleNamespace(scan_scale=None,overlap=.5)))
    volume = zarr.open_array(work/"plane.zarr", mode="r")
    module = load_module(source)
    start = perf_counter()
    train, heldout = module.find_segment_unlabeled_patches(segment,lambda name,scale:volume)
    elapsed = perf_counter()-start
    boxes = json.dumps([p.bbox for p in train], separators=(",",":")).encode()
    return dict(seconds=elapsed, count=len(train), heldout=len(heldout), ordered_bbox_sha256=hashlib.sha256(boxes).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream",type=Path)
    parser.add_argument("patched",type=Path)
    parser.add_argument("--worker", type=Path)
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(measure(args.upstream,args.worker)))
    else:
        assert hashlib.sha256(args.upstream.read_bytes()).hexdigest()=="439b6b0c25c1d496475399651ac3f3cbb03e96d6a76fdb4f2556f5acc4d03896"
        # Use an isolated cache because fetch_roi keys chunks by coordinates.
        import os
        script=Path(__file__).resolve()
        sources=[p.resolve() for p in (args.upstream,args.patched)]
        with tempfile.TemporaryDirectory() as dirname:
            work=Path(dirname)
            os.chdir(work)
            result=prepare(work)
            runs=[[],[]]
            for order in ((0,1),(1,0),(0,1),(1,0)):
                for i in order:
                    output=subprocess.check_output([sys.executable,script,sources[i],sources[i],"--worker",work],text=True)
                    runs[i].append(json.loads(output.splitlines()[-1]))
            identities={(r["count"],r["heldout"],r["ordered_bbox_sha256"]) for group in runs for r in group}
            assert len(identities)==1 and runs[0][0]["count"]>0 and runs[0][0]["heldout"]==0
            result["upstream"],result["patched"]=runs
            result["median_speedup"]=median(r["seconds"] for r in runs[0])/median(r["seconds"] for r in runs[1])
            print(json.dumps(result, indent=2))
