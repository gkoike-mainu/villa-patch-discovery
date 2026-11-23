"""Check full ordered discovery results and empty-plane errors on seed 4721."""
import argparse
import contextlib
from dataclasses import make_dataclass
import io
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import numpy as np

def load(path):
    m=ModuleType("discovery")
    exec(compile(path.read_bytes(),str(path),"exec"),m.__dict__)
    return m

def output(impl,segment,volumes):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            train,valid=impl.find_segment_unlabeled_patches(segment,lambda name,level:volumes[name,level])
        return ([ (p.bbox,p.is_unlabeled) for p in train], [p.bbox for p in valid])
    except ValueError as error:
        return ("error",str(error))

def compare(upstream,candidate):
    for name in ("vesuvius","vesuvius.ink_detection","vesuvius.ink_detection.types"):
        sys.modules[name]=ModuleType(name)
    types=sys.modules["vesuvius.ink_detection.types"]
    types.Patch=make_dataclass("Patch",["segment","bbox",("is_unlabeled",bool,False)]);types.Segment=object
    impls=list(map(load,(upstream,candidate)))
    rng=np.random.default_rng(4721)
    for n in range(320):
        h,w=int(rng.integers(1,60)),int(rng.integers(1,78))
        dims=(3,h,w)
        img=rng.choice([-3.,0.,0.,.3,np.nan],size=dims)
        if n%5==0:img[:]=0
        elif n%5==1:img[rng.random(dims)<.96]=0
        if n%3==0:img=np.ascontiguousarray(img!=0,dtype=np.uint8)
        sup,val=('sup' if n&1 else None),('val' if n&2 else None)
        scan=1 if n%4==0 else 0
        v={('image',scan):img,('sup',scan):rng.random(dims)<.005,('val',scan):rng.random(dims)<.007}
        if scan:v['image',0]=np.broadcast_to(0,(5,h*2+1,w*2-1))
        cfg=SimpleNamespace(scan_scale=scan,overlap=(.01,.25,.5,.74,1.)[n%5])
        seg=SimpleNamespace(image_volume='image',scale=0,patch_size=(3,int(rng.integers(1,57)),int(rng.integers(1,66))),supervision_mask=sup,validation_mask=val,data_config=SimpleNamespace(patch_finding=cfg))
        assert output(impls[0],seg,v)==output(impls[1],seg,v),n
    print("320 equivalence cases passed")
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('upstream',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args()
    compare(a.upstream,a.candidate)
