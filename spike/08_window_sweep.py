"""Measure how the window shape and the frame rate change retrieval rank.

The video is the 30 s testvideo_nolabels.mp4. It holds three 10 s blocks:
0-10 s sunny, 10-20 s tuesday, 20-30 s birthday. Four queries have a known
true block, so the rank of the first overlapping window is the score.

Sweep axes:
  frames per window: the fps argument (1, 2, or 4)
  window and stride: (4 s, 2 s) and (2 s, 1 s)
  input shape: separate JPEG frames, or the native mp4 video clip

Run:
    d:/MLProjelerim/.venv_ml/Scripts/python.exe 08_window_sweep.py
"""
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
VIDEO = SPIKE / "testvideo_nolabels.mp4"
WORK = SPIKE / "sweep_work"

# name, query, true block as (start, end)
QUERIES = [
    ("birthday", "birthday party", (20.0, 30.0)),
    ("sunny", "sunny warm weather", (0.0, 10.0)),
    ("tuesday", "quiet tuesday morning", (10.0, 20.0)),
    ("blue", "blue", (10.0, 20.0)),
]

# label, window, stride, fps (None means the native video clip)
CONFIGS = [
    ("frames f1 w4", 4.0, 2.0, 1),
    ("frames f2 w4", 4.0, 2.0, 2),
    ("frames f4 w4", 4.0, 2.0, 4),
    ("video       w4", 4.0, 2.0, None),
    ("frames f1 w2", 2.0, 1.0, 1),
    ("frames f2 w2", 2.0, 1.0, 2),
    ("video       w2", 2.0, 1.0, None),
]


def probe_duration(video):
    out = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def run(args):
    subprocess.run([str(FFMPEG)] + args, check=True, capture_output=True)


def cut(dir_, start, win, fps, video):
    """Return (clip_mp4_or_None, wav, frame_list)."""
    dir_.mkdir(parents=True, exist_ok=True)
    span = ["-y", "-ss", str(start), "-t", str(win), "-i", str(video)]
    wav = dir_ / "clip.wav"
    run(span + ["-vn", "-ac", "1", "-ar", "16000", str(wav)])
    frames = []
    mp4 = None
    if fps is None:
        mp4 = dir_ / "clip.mp4"
        run(span + ["-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(mp4)])
    else:
        run(span + ["-vf", "fps=%d" % fps, "-q:v", "2", str(dir_ / "f%02d.jpg")])
        frames = sorted(str(p) for p in dir_.glob("f*.jpg"))
    return (str(mp4) if mp4 else None), str(wav), frames


def embed_window(model, mp4, wav, frames):
    """Return (unit_vector, shape_name). Try audio first, then drop it."""
    payloads = []
    if frames:
        payloads.append(("frames+audio", {"image": frames, "audio": wav}))
        payloads.append(("frames", {"image": frames}))
    else:
        payloads.append(("video+audio", {"video": mp4, "audio": wav}))
        payloads.append(("video", {"video": mp4}))
    last = ""
    for name, payload in payloads:
        try:
            v = np.asarray(model.encode(payload), dtype=np.float32).reshape(-1)
            return v / (np.linalg.norm(v) + 1e-9), name
        except Exception as exc:
            last = "%s: %s" % (type(exc).__name__, str(exc)[:80])
    raise RuntimeError("no payload worked: " + last)


def index(model, win, stride, fps, duration):
    vecs, spans, shapes = [], [], []
    k, start = 0, 0.0
    while start + win <= duration + 1e-6:
        d = WORK / ("w%.0f_s%.0f_f%s" % (win, stride, fps)) / ("c" + str(k))
        mp4, wav, frames = cut(d, start, win, fps, VIDEO)
        v, shape = embed_window(model, mp4, wav, frames)
        vecs.append(v)
        spans.append((start, min(start + win, duration)))
        shapes.append(shape)
        start += stride
        k += 1
    return np.stack(vecs), spans, shapes


def score(vecs, spans, qvec):
    scores = vecs @ qvec
    order = np.argsort(-scores)
    return scores, list(order)


def rank_of_hit(order, spans, block):
    for rank, idx in enumerate(order, 1):
        s, e = spans[idx]
        if s < block[1] and e > block[0]:
            return rank, idx
    return None, None


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(VIDEO)
    print("video %s  duration %.1f s" % (VIDEO.name, duration))

    model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16})
    qvecs = {}
    for name, text, _ in QUERIES:
        q = np.asarray(model.encode(text, prompt_name="SearchQuery"),
                       dtype=np.float32).reshape(-1)
        qvecs[name] = q / (np.linalg.norm(q) + 1e-9)

    summary = []
    for label, win, stride, fps in CONFIGS:
        vecs, spans, shapes = index(model, win, stride, fps, duration)
        shape = shapes[0]
        print("\n=== %s  (%d windows, shape %s) ===" % (label, len(spans), shape))
        rows = []
        for name, text, block in QUERIES:
            scores, order = score(vecs, spans, qvecs[name])
            rank, idx = rank_of_hit(order, spans, block)
            top = order[0]
            ts, te = spans[top]
            true_scores = [float(scores[i]) for i in range(len(spans))
                           if spans[i][0] < block[1] and spans[i][1] > block[0]]
            false_scores = [float(scores[i]) for i in range(len(spans))
                            if not (spans[i][0] < block[1] and spans[i][1] > block[0])]
            best_true = max(true_scores) if true_scores else float("nan")
            best_false = max(false_scores) if false_scores else float("nan")
            rows.append(rank)
            print("  %-8s top %.3f %5.1f-%5.1f | hit rank %s | true %.3f false %.3f margin %+.3f"
                  % (name, scores[top], ts, te, rank, best_true, best_false,
                     best_true - best_false))
        hits = sum(1 for r in rows if r == 1)
        summary.append((label, hits, rows))

    print("\n=== summary ===")
    print("%-14s %-8s %s" % ("config", "hits@1", "ranks (birthday,sunny,tuesday,blue)"))
    for label, hits, rows in summary:
        print("%-14s %d/4      %s" % (label, hits, rows))


if __name__ == "__main__":
    main()
