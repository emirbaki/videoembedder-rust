"""Cut testvideo.mp4 into windows, embed each window, embed the queries.

Tries three input shapes per window and keeps the first one that works, so the
run also reports what this version of sentence-transformers accepts.
Writes spike/embeddings.json
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("D:/MLProjelerim/videoembedder-rust")
FFMPEG = ROOT / "tools/dl/ff/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"

# torchcodec needs the FFmpeg shared libraries before it loads.
SHARED = ROOT / "tools/dl/ffs/ffmpeg-master-latest-win64-gpl-shared/bin"
if SHARED.is_dir():
    os.add_dll_directory(str(SHARED))
    os.environ["PATH"] = str(SHARED) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

WORK = ROOT / "spike/work"
OUT = ROOT / "spike"
VIDEO_NAME = sys.argv[1] if len(sys.argv) > 1 else "testvideo.mp4"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "embeddings"
VIDEO = OUT / VIDEO_NAME

WIN, STRIDE, DUR = 6.0, 3.0, 30.0
QUERIES = ["birthday party", "sunny warm weather", "quiet tuesday morning"]


def cut(k, t0):
    d = WORK / ("w" + str(k))
    d.mkdir(parents=True, exist_ok=True)
    ss = ["-y", "-ss", str(t0), "-t", str(WIN), "-i", str(VIDEO)]
    subprocess.run([str(FFMPEG)] + ss + ["-c:v", "libx264", "-preset", "ultrafast",
                   "-pix_fmt", "yuv420p", str(d / "clip.mp4")], check=True, capture_output=True)
    subprocess.run([str(FFMPEG)] + ss + ["-vn", "-ac", "1", "-ar", "16000",
                   str(d / "clip.wav")], check=True, capture_output=True)
    subprocess.run([str(FFMPEG)] + ss + ["-vf", "fps=1", "-q:v", "2",
                   str(d / "f%02d.jpg")], check=True, capture_output=True)
    return str(d / "clip.mp4"), str(d / "clip.wav"), [str(p) for p in sorted(d.glob("f*.jpg"))]


def first_ok(model, payloads):
    """Return (vector, shape_name, error_text)."""
    err = ""
    for name, payload in payloads:
        try:
            v = model.encode(payload)
            return np.asarray(v, dtype=np.float32).reshape(-1), name, ""
        except Exception as exc:
            err = name + ": " + type(exc).__name__ + " " + str(exc)[:160]
    return None, "none", err


def main():
    t0 = time.perf_counter()
    model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16})
    print("model loaded in %.1f s, device cuda, dim %d" % (time.perf_counter() - t0,
                                                           model.get_sentence_embedding_dimension()))

    qvecs = model.encode(QUERIES, prompt_name="SearchQuery", normalize_embeddings=True)
    qvecs = np.asarray(qvecs, dtype=np.float32)

    rows = []
    store = []
    for k, start in enumerate(np.arange(0.0, DUR - WIN + 0.001, STRIDE)):
        mp4, wav, frames = cut(k, start)
        payloads = [
            ("video+audio", {"video": mp4, "audio": wav}),
            ("audio", {"audio": wav}),
            ("video", {"video": mp4}),
            ("frames+audio", {"image": frames, "audio": wav}),
            ("frames", {"image": frames}),
        ]
        t1 = time.perf_counter()
        vec, shape, err = first_ok(model, payloads)
        dt = time.perf_counter() - t1
        sims = None
        if vec is not None:
            v = vec / (np.linalg.norm(vec) + 1e-9)
            store.append((float(start), float(start + WIN), v))
            sims = [float(np.dot(v, q / (np.linalg.norm(q) + 1e-9))) for q in qvecs]
        rows.append({"k": k, "start": float(start), "end": float(start + WIN),
                     "shape": shape, "seconds": round(dt, 2), "sims": sims, "error": err})
        print("w%d %4.1f-%4.1f s | %-12s %5.2f s | %s" % (
            k, start, start + WIN, shape, dt,
            " ".join("%.3f" % s for s in sims) if sims else err[:70]))

    if store:
        np.savez(OUT / (PREFIX + ".npz"),
                 vecs=np.stack([s[2] for s in store]),
                 starts=np.array([s[0] for s in store]),
                 ends=np.array([s[1] for s in store]))
        print("saved", len(store), "normalized window vectors")
    (OUT / (PREFIX + ".json")).write_text(json.dumps(
        {"video": VIDEO_NAME, "queries": QUERIES, "windows": rows}, indent=1),
        encoding="utf-8")
    print("wrote", OUT / (PREFIX + ".json"))


if __name__ == "__main__":
    main()
