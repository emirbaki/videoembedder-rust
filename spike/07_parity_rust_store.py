"""Compare the Rust store vectors with the Python index vectors.

The store file starts with magic "VEI1". Records hold a label, a time range,
and one float32 vector. The Python index sits in index_testvideo_nolabels.npz.
Only windows with the same time range are comparable.
"""

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STORE = HERE.parent / "videoembed" / "store" / "index.vei"
NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index_testvideo_nolabels.npz"


def read_store(path):
    """Parse the store file into a list of (kind, label, start, end, vec)."""
    raw = path.read_bytes()
    if raw[:4] != b"VEI1":
        raise SystemExit("bad magic in the store file")
    dim, count = struct.unpack_from("<II", raw, 4)
    at = 12
    out = []
    for _ in range(count):
        kind = raw[at]
        (label_len,) = struct.unpack_from("<I", raw, at + 1)
        at += 5
        label = raw[at : at + label_len].decode("utf-8")
        at += label_len
        start, end = struct.unpack_from("<ff", raw, at)
        at += 8
        vec = np.frombuffer(raw, dtype="<f4", count=dim, offset=at).astype(np.float32)
        at += 4 * dim
        out.append((kind, label, start, end, vec))
    return dim, out


def main():
    dim, records = read_store(STORE)
    print(f"store dim {dim}, {len(records)} records")
    data = np.load(NPZ)
    py_vecs, py_starts, py_ends = data["vecs"], data["starts"], data["ends"]
    print(f"python dim {py_vecs.shape[1]}, {len(py_vecs)} records")

    for _kind, _label, start, end, vec in records:
        hit = np.where((np.abs(py_starts - start) < 0.01) & (np.abs(py_ends - end) < 0.01))[0]
        if len(hit) == 0:
            print(f"  {start:5.1f}-{end:5.1f}  no python window with this range")
            continue
        other = py_vecs[hit[0]]
        cos = float(vec / np.linalg.norm(vec) @ (other / np.linalg.norm(other)))
        print(f"  {start:5.1f}-{end:5.1f}  cosine {cos:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
