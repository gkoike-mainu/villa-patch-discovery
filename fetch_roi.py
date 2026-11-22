"""Extract the central plane of a 1024 x 1024 public OME-Zarr crop."""
import concurrent.futures
import urllib.request
import urllib.error
import time
from pathlib import Path
import numpy as np

ROOT = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231022170901/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/3"
SIDE = 128
PLANE = 54
FIRST = PLANE * SIDE**2
LAST = FIRST + SIDE**2 - 1

def fetch(cell):
    row, col = cell
    key = f"{ROOT}/0/{row}/{col}"
    req = urllib.request.Request(key, headers={"Range": f"bytes={FIRST}-{LAST}"})
    cache = Path("chunks") / f"{row}_{col}.bin"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        return row, col, np.frombuffer(cache.read_bytes(), dtype=np.uint8).reshape(SIDE, SIDE)
    for attempt in range(4):
        try:
            response = urllib.request.urlopen(req, timeout=35)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return row, col, np.zeros((SIDE,SIDE), dtype=np.uint8)
            if exc.code not in (429,500,502,503,504) or attempt == 3:
                raise
            time.sleep(2 ** attempt)
    with response:
        raw = response.read()
        if response.status == 200 and len(raw) == 109 * SIDE**2:
            raw = raw[FIRST:LAST+1]
        elif response.status != 206 or not response.headers["Content-Range"].startswith(f"bytes {FIRST}-{LAST}/"):
            raise ValueError(f"Unexpected response for {key}")
    cache.write_bytes(raw)
    return row, col, np.frombuffer(raw, dtype=np.uint8).reshape(SIDE, SIDE)

if __name__ == "__main__":
    image = np.empty((1,1024,1024),dtype=np.uint8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for row,col,data in pool.map(fetch,[(r,c) for r in range(6,14) for c in range(52,60)]):
            y,x=(row-6)*SIDE,(col-52)*SIDE
            image[0,y:y+SIDE,x:x+SIDE]=data
    np.save("PHercParis4_20231022170901_level3_z54.npy",image)
    print(image.shape, np.count_nonzero(image))
