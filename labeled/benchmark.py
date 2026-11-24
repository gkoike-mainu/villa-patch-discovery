"""Time real labelled training-patch discovery and compare the ordered coordinate lists."""
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace, ModuleType
from time import perf_counter
import hashlib, json, resource, sys
import numpy as np

def convert(root, step):
    import tifffile
    for name in ('inklabels','supervision'):
        with tifffile.TiffFile(root/(name+'.tif')) as tif:
            page,=tif.pages
            if page.dtype != np.uint8 or page.shape != (32249,51380):
                raise ValueError('Unexpected official TIFF geometry')
            output=np.lib.format.open_memmap(root/(name+'.npy'),mode='w+',dtype='u1',
                shape=tuple((n+step-1)//step for n in page.shape))
            for tile,(_,_,y,x,_),_ in page.segments(maxworkers=4):
                if y%step or x%step:raise ValueError('Unaligned tile origins')
                data=tile[0,::step,::step,0]
                dest=output[y//step:y//step+data.shape[0],x//step:x//step+data.shape[1]]
                dest[:]=data[:dest.shape[0],:dest.shape[1]]
            output.flush()

def run(path,root,patch):
    for name in ('vesuvius','vesuvius.ink_detection','vesuvius.ink_detection.types'):
        sys.modules[name]=ModuleType(name)
    types=sys.modules['vesuvius.ink_detection.types']
    types.Patch=make_dataclass('Patch',['segment','bbox',('is_unlabeled',bool,False),
        ('is_validation',bool,False),('supervision_mask_override',object,None)])
    types.Segment=object
    mod=ModuleType('discovery')
    exec(compile(Path(path).read_bytes(),path,'exec'),mod.__dict__)
    supervision=np.load(root/'supervision.npy',mmap_mode='r')[None,:,:]
    ink=np.load(root/'inklabels.npy',mmap_mode='r')[None,:,:]
    image=SimpleNamespace(shape=ink.shape)
    config=SimpleNamespace(patch_finding=SimpleNamespace(scan_scale=None,overlap=.5,min_labeled_coverage=.06))
    segment=SimpleNamespace(inklabels='ink',image_volume='image',scale=0,patch_size=(1,patch,patch),
        supervision_mask='supervision',validation_mask=None,data_config=config)
    volumes={'supervision':supervision,'ink':ink,'image':image}
    start=perf_counter()
    train,validation=mod.find_segment_patches(segment,lambda name,scale:volumes[name])
    seconds=perf_counter()-start
    boxes=[(p.bbox,p.is_validation) for p in train+validation]
    print(json.dumps(dict(seconds=seconds,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      boxes=len(boxes),hash=hashlib.sha256(repr(boxes).encode()).hexdigest(),shape=ink.shape)))

if __name__=='__main__':
    root=Path(sys.argv[2])
    if sys.argv[1]=='convert':convert(root,int(sys.argv[3]) if len(sys.argv)>3 else 4)
    else:run(sys.argv[1],root,int(sys.argv[3]) if len(sys.argv)>3 else 128)
