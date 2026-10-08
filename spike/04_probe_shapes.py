"""Probe which multimodal input shapes this build accepts, and why the rest fail.

Tries every shape on one window and prints the exact exception for each one.
"""
import os
from pathlib import Path

ROOT = Path("D:/MLProjelerim/videoembedder-rust")
WORK = ROOT / "spike/work"

# torchcodec needs the FFmpeg shared libraries. On Windows no PATH lookup happens
# for extension modules, so register the directory before torchcodec loads.
SHARED = ROOT / "tools/dl/ffs/ffmpeg-master-latest-win64-gpl-shared/bin"
if SHARED.is_dir():
    os.add_dll_directory(str(SHARED))
    os.environ["PATH"] = str(SHARED) + os.pathsep + os.environ.get("PATH", "")

import torch
from sentence_transformers import SentenceTransformer

for mod in ("soundfile", "librosa", "av", "torchvision", "decord", "PIL"):
    try:
        __import__(mod)
        print("present :", mod)
    except Exception as exc:
        print("MISSING :", mod, "->", type(exc).__name__)

print()
model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                           model_kwargs={"torch_dtype": torch.bfloat16})

d = WORK / "w6"
frames = [str(p) for p in sorted(d.glob("f*.jpg"))]
cases = [
    ("video+audio", {"video": str(d / "clip.mp4"), "audio": str(d / "clip.wav")}),
    ("video-file", {"video": str(d / "clip.mp4")}),
    ("audio-file", {"audio": str(d / "clip.wav")}),
    ("frames+audio", {"image": frames, "audio": str(d / "clip.wav")}),
    ("frames", {"image": frames}),
    ("text+placeholder", {"text": "a short clip <|video|>", "video": str(d / "clip.mp4")}),
]
for name, payload in cases:
    try:
        v = model.encode(payload)
        import numpy as np
        print("%-18s OK    dim=%d" % (name, np.asarray(v).reshape(-1).shape[0]))
    except Exception as exc:
        msg = (type(exc).__name__ + ": " + str(exc)).replace("\n", " ")[:220]
        print("%-18s FAIL  %s" % (name, msg))
