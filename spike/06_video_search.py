"""Index one video and search it by text. Shows the matching time range.

Usage:
    python 06_video_search.py index spike/testvideo_nolabels.mp4
    python 06_video_search.py search "birthday party"

The search command opens a prompt. Type a query and press Enter. Type quit to stop.

The index stores one vector per time window. Window vectors use the video and
the audio track together. The first payload that works wins, so a window with
broken audio falls back to video only.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("D:/MLProjelerim/videoembedder-rust")
FFMPEG = ROOT / "tools/dl/ff/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
FFPROBE = FFMPEG.with_name("ffprobe.exe")

# torchcodec needs the FFmpeg shared libraries before it loads.
SHARED = ROOT / "tools/dl/ffs/ffmpeg-master-latest-win64-gpl-shared/bin"
if SHARED.is_dir():
    os.add_dll_directory(str(SHARED))
    os.environ["PATH"] = str(SHARED) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SPIKE = ROOT / "spike"
WORK = SPIKE / "search_work"
WIN = float(os.environ.get("VEI_WIN", "6.0"))
STRIDE = float(os.environ.get("VEI_STRIDE", "3.0"))
# A suffix keeps a second setting from overwriting the first index.
SUFFIX = os.environ.get("VEI_SUFFIX", "")


def load_model():
    return SentenceTransformer("google/embeddinggemma-2", device="cuda",
                               model_kwargs={"torch_dtype": torch.bfloat16})


def probe_duration(video):
    out = subprocess.run(
        [str(FFPROBE), "-v", "error",
         "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def cut(k, t0, video):
    d = WORK / ("w" + str(k))
    d.mkdir(parents=True, exist_ok=True)
    span = ["-y", "-ss", str(t0), "-t", str(WIN), "-i", str(video)]
    subprocess.run([str(FFMPEG)] + span + ["-c:v", "libx264", "-preset", "ultrafast",
                   "-pix_fmt", "yuv420p", str(d / "clip.mp4")],
                   check=True, capture_output=True)
    subprocess.run([str(FFMPEG)] + span + ["-vn", "-ac", "1", "-ar", "16000",
                   str(d / "clip.wav")], check=True, capture_output=True)
    return str(d / "clip.mp4"), str(d / "clip.wav")


def embed_window(model, mp4, wav):
    payloads = [("video+audio", {"video": mp4, "audio": wav}),
                ("video", {"video": mp4}),
                ("audio", {"audio": wav})]
    for name, payload in payloads:
        try:
            v = np.asarray(model.encode(payload), dtype=np.float32).reshape(-1)
            return v / (np.linalg.norm(v) + 1e-9), name
        except Exception:
            continue
    raise RuntimeError("no payload worked")


def index(video):
    video = Path(video)
    model = load_model()
    duration = probe_duration(video)
    print("video %s  duration %.1f s" % (video.name, duration))

    win = JSON_PATH(video)
    vecs, starts, ends, shapes = [], [], [], []
    k = 0
    start = 0.0
    while start + WIN <= duration + 1e-6:
        mp4, wav = cut(k, start, video)
        v, shape = embed_window(model, mp4, wav)
        vecs.append(v)
        starts.append(start)
        ends.append(min(start + WIN, duration))
        shapes.append(shape)
        print("  w%d %5.1f-%5.1f s  %-11s" % (k, start, start + WIN, shape))
        start += STRIDE
        k += 1

    np.savez(win.with_suffix(".npz"), vecs=np.stack(vecs),
             starts=np.array(starts), ends=np.array(ends))
    win.write_text(json.dumps({"video": str(video), "shapes": shapes}), encoding="utf-8")
    print("indexed %d windows -> %s" % (len(vecs), win.with_suffix(".npz")))


def JSON_PATH(video):
    return SPIKE / ("index_" + Path(video).stem + SUFFIX + ".json")


def search(video):
    video = Path(video)
    win = JSON_PATH(video)
    if not win.with_suffix(".npz").exists():
        print("no index for %s. Run: index %s" % (video, video))
        return
    d = np.load(win.with_suffix(".npz"))
    vecs, starts, ends = d["vecs"], d["starts"], d["ends"]

    model = load_model()
    while True:
        try:
            query = input("query> ").strip()
        except EOFError:
            break
        if not query or query in ("quit", "exit"):
            break
        q = np.asarray(model.encode(query, prompt_name="SearchQuery"), dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = vecs @ q
        order = np.argsort(-scores)
        print()
        for rank, idx in enumerate(order[:5], 1):
            bar = "#" * max(0, int(round(float(scores[idx]) * 40)))
            print("  %d. %5.1f-%5.1f s   %.3f  %s"
                  % (rank, starts[idx], ends[idx], scores[idx], bar))
        print()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    cmd, video = sys.argv[1], sys.argv[2]
    WORK.mkdir(parents=True, exist_ok=True)
    if cmd == "index":
        index(video)
    elif cmd == "search":
        search(video)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
