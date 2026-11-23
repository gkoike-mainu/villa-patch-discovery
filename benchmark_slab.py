"""Compare ordered patch discovery and subsequent three-layer Zarr reads on PHercParis4."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import resource
from statistics import median
import subprocess
import sys
import time
from dataclasses import make_dataclass
from types import ModuleType, SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import zarr

ROOT = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231005123336/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/5"
UPSTREAM = "439b6b0c25c1d496475399651ac3f3cbb03e96d6a76fdb4f2556f5acc4d03896"

def prepare(work):
    with urlopen(ROOT + "/.zarray", timeout=35) as res:
        meta=json.load(res)
    assert (meta["shape"],meta["chunks"],meta["dtype"],meta["compressor"],meta["order"],meta["fill_value"]) == ([109,1090,3040],[109,128,128],"|u1",None,"C",0)
    h,w=meta["shape"][1:]
    plane=np.zeros((3,h,w),dtype="u1")
    def fetch(cell):
        r,c=cell
        start,end=53*128*128,56*128*128-1
        req=Request(f"{ROOT}/0/{r}/{c}",headers={"Range":f"bytes={start}-{end}"})
        for attempt in range(4):
            try:
                with urlopen(req,timeout=45) as response:
                    raw=response.read()
                    if response.status==200 and len(raw)==109*128**2:
                        raw=raw[start:end+1]
                    elif response.status!=206 or not response.headers.get("Content-Range","").startswith(f"bytes {start}-{end}/"):
                        raise ValueError("Unexpected range")
                return r,c,np.frombuffer(raw,dtype="u1").reshape(3,128,128)
            except HTTPError as err:
                if err.code==404: return r,c,None
                if err.code not in (429,500,502,503,504) or attempt==3: raise
                time.sleep(2**attempt)
    cells=[(r,c) for r in range((h+127)//128) for c in range((w+127)//128)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for r,c,block in pool.map(fetch,cells):
            if block is not None:
                target=plane[:,r*128:(r+1)*128,c*128:(c+1)*128]
                target[:]=block[:,:target.shape[1],:target.shape[2]]
    arr=zarr.open_array(work/"slab.zarr",mode="w",shape=plane.shape,chunks=(3,128,128),dtype="u1",zarr_format=2)
    arr[:]=plane
    return dict(source=ROOT,source_planes=[53,54,55],shape=list(plane.shape),sha256=hashlib.sha256(plane.tobytes()).hexdigest(),middle_nonzero=int(np.count_nonzero(plane[1])))

def measure(source,work):
    for name in ("vesuvius","vesuvius.ink_detection","vesuvius.ink_detection.types"):
        sys.modules[name]=ModuleType(name)
    types=sys.modules["vesuvius.ink_detection.types"]
    types.Patch=make_dataclass("Patch",["segment","bbox",("is_unlabeled",bool,False)])
    types.Segment=object
    module=ModuleType("discovery")
    exec(compile(source.read_bytes(),str(source),"exec"),module.__dict__)
    seg=SimpleNamespace(image_volume="image",scale=0,patch_size=(3,256,256),supervision_mask=None,validation_mask=None,data_config=SimpleNamespace(patch_finding=SimpleNamespace(scan_scale=None,overlap=.5)))
    t0=time.perf_counter()
    slab=zarr.open_array(work/"slab.zarr",mode="r")
    train,val=module.find_segment_unlabeled_patches(seg,lambda name,scale:slab)
    t1=time.perf_counter()
    records=[]
    for patch in train:
        z,y,x,z1,y1,x1=patch.bbox
        vol=np.asarray(slab[z:z1,y:y1,x:x1])
        records.append([patch.bbox,patch.is_unlabeled,list(vol.shape),int(vol.sum(dtype=np.uint64)),int(np.count_nonzero(vol))])
    t2=time.perf_counter()
    return dict(discovery_seconds=t1-t0,workflow_seconds=t2-t0,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,windows=len(records),validation=len(val),digest=hashlib.sha256(json.dumps(records,separators=(",",":")).encode()).hexdigest())

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("sources",nargs="+",type=Path)
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args();args.work.mkdir(parents=True,exist_ok=True)
    if args.worker:
        print(json.dumps(measure(args.sources[0],args.work)))
    else:
        assert len(args.sources)==3 and hashlib.sha256(args.sources[0].read_bytes()).hexdigest()==UPSTREAM
        result=prepare(args.work)
        names=["upstream","previous","candidate"];runs={n:[] for n in names}
        for order in ((0,1,2),(2,1,0),(1,2,0),(0,2,1)):
            for i in order:
                raw=subprocess.check_output([sys.executable,__file__,args.work,args.sources[i],"--worker"],text=True)
                runs[names[i]].append(json.loads(raw.splitlines()[-1]))
        assert len({(r["digest"],r["windows"],r["validation"]) for group in runs.values() for r in group})==1
        assert runs["candidate"][0]["windows"]>0
        result.update(runs=runs,medians={k:{metric:median(r[metric] for r in group) for metric in ("discovery_seconds","workflow_seconds","peak_rss_kib")} for k,group in runs.items()})
        print(json.dumps(result,indent=2))
