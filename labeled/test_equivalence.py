"""Exercise the labelled and unlabelled paths under different masks and scale settings."""
from dataclasses import make_dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import numpy as np

def run(old_path,new_path):
    for name in ('vesuvius','vesuvius.ink_detection','vesuvius.ink_detection.types'):
        sys.modules[name]=ModuleType(name)
    t=sys.modules['vesuvius.ink_detection.types']
    t.Patch=make_dataclass('Patch',['segment','bbox',('is_unlabeled',bool,False),
        ('is_validation',bool,False),('supervision_mask_override',object,None)])
    t.Segment=object
    mods=[]
    for path in (old_path,new_path):
        m=ModuleType('patch_finding')
        exec(compile(Path(path).read_bytes(),path,'exec'),m.__dict__)
        mods.append(m)
    rng=np.random.default_rng(1216)
    for i in range(240):
        h,w=rng.integers(2,95,size=2)
        stride=int(rng.integers(1,10))
        height,width=int(rng.integers(1,30)),int(rng.integers(1,30))
        shape=(3,h,w)
        objects={k:rng.choice([0,1,2],size=shape,p=[.88,.095,.025]).astype('u1') for k in ('scan','supervision','ink','validation')}
        if i%5==0:objects['supervision'][:]=0
        if i%6==0:objects['scan'][:]=0
        config=SimpleNamespace(patch_finding=SimpleNamespace(scan_scale=None,overlap=stride/height,min_labeled_coverage=.15))
        s=SimpleNamespace(image_volume='scan',scale=0,patch_size=(3,height,width),supervision_mask='supervision',inklabels='ink',
                           validation_mask='validation' if i%3 else None,data_config=config)
        def output(module,name):
            try:
                p=getattr(module,name)(s,lambda key,scale:objects[key])
            except ValueError as e:return ('error',str(e))
            return [[(x.bbox,x.is_validation,x.is_unlabeled) for x in group] for group in p]
        for fn in ['find_segment_patches','find_segment_unlabeled_patches']:
            assert output(mods[0],fn)==output(mods[1],fn),(i,fn)
    print('480 patch-list comparisons passed.')
if __name__=='__main__':run(*sys.argv[1:])
