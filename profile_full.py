"""Measure full-depth maximum-intensity projection of each discovered patch."""
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from dataclasses import make_dataclass
from types import ModuleType, SimpleNamespace
import numpy as np

def measure(path, volume):
    for name in ('vesuvius','vesuvius.ink_detection','vesuvius.ink_detection.types'):
        sys.modules[name]=ModuleType(name)
    types=sys.modules['vesuvius.ink_detection.types']
    types.Patch=make_dataclass('Patch',['segment','bbox',('is_unlabeled',bool,False)])
    types.Segment=object
    module=ModuleType('discovery')
    exec(compile(Path(path).read_bytes(),path,'exec'),module.__dict__)
    start=perf_counter()
    scan=np.load(volume,mmap_mode='r')
    segment=SimpleNamespace(image_volume='image',scale=0,patch_size=(scan.shape[0],256,256),supervision_mask=None,validation_mask=None,data_config=SimpleNamespace(patch_finding=SimpleNamespace(scan_scale=None,overlap=.5)))
    train,valid=module.find_segment_unlabeled_patches(segment,lambda name,scale:scan)
    found=perf_counter()
    digest=hashlib.sha256()
    assert train and not valid
    for p in train:
        z,y,x,d,h,w=p.bbox
        slab=scan[z:d,y:h,x:w]
        assert slab.shape[0]==scan.shape[0]
        digest.update(repr(p.bbox).encode())
        digest.update(slab.max(axis=0).tobytes())
    return dict(seconds=perf_counter()-start,discovery=found-start,boxes=len(train),digest=digest.hexdigest(),shape=scan.shape)
if __name__=='__main__':print(json.dumps(measure(*sys.argv[1:])))
