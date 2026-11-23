"""Compare discovery and full-depth patch projection on one complete surface volume."""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError
import numpy as np

ROOT = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231022170901/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/5'
def prepare(folder):
    folder.mkdir(exist_ok=True)
    with urlopen(ROOT+'/.zarray',timeout=60) as response:
        metadata=json.load(response)
    assert metadata['dtype']=='|u1' and metadata['compressor'] is None and metadata['fill_value']==0 and metadata['order']=='C'
    d,h,w=metadata['shape']
    assert metadata['chunks']==[d,128,128]
    def chunk(pair):
        i,j=pair
        path=folder/f'{i}_{j}.bin'
        if path.exists() and path.stat().st_size==d*128**2:return
        for attempt in range(4):
            try:
                with urlopen(f'{ROOT}/0/{i}/{j}',timeout=60) as response: raw=response.read()
                break
            except HTTPError as err:
                if err.code==404: raw=bytes(d*128**2);break
                if attempt==3:raise
            except OSError:
                if attempt==3:raise
        if len(raw)!=d*128**2:raise ValueError('unexpected chunk length')
        path.write_bytes(raw)
    pairs=[(i,j) for i in range((h+127)//128) for j in range((w+127)//128)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(chunk,pairs))
    vol=np.lib.format.open_memmap(folder/'volume.npy','w+',dtype='u1',shape=(d,h,w))
    for i,j in pairs:
        part=np.fromfile(folder/f'{i}_{j}.bin',dtype='u1').reshape(d,128,128)
        dest=vol[:,i*128:(i+1)*128,j*128:(j+1)*128]
        dest[:]=part[:,:dest.shape[1],:dest.shape[2]]
    vol.flush()
if __name__=='__main__':prepare(Path('scan'))
