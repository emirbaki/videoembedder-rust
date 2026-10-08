"""Probe the coarse-to-fine search. One coarse pass, then one fine pass.

A coarse pass indexes long windows (4 s, 2 s stride). For one query it takes
the top coarse hit and re-embeds only that window at a short window (1 s,
0.5 s stride). The fine pass should land inside the coarse hit and give a
narrower range.

Two videos have a known truth:
  testvideo_nolabels.mp4  30 s, three 10 s blocks
  the football clip        9.3 s, woman first, then crying players

Run:
    d:/MLProjelerim/.venv_ml/Scripts/python.exe 09_coarse_fine.py
"""
import os
import subprocess
from pathlib import Path

ROOT = Path("D:/MLProjelerim/videoembedder-rust")
FFMPEG = ROOT / "tools/dl/ff/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
FFPROBE = FFMPEG.with_name("ffprobe.exe")

SHARED = ROOT / "tools/dl/ffs/ffmpeg-master-latest-win64-gpl-shared/bin"
if SHARED.is_dir():
    os.add_dll_directory(str(SHARED))
    os.environ["PATH"] = str(SHARED) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SPIKE = ROOT / "spike"
WORK = SPIKE / "sweep_work"

COARSE_W, COARSE_S = 4.0, 2.0
FINE_W, FINE_S = 1.0, 0.5
FPS = 1
FINE_FPS = 4

FOOTBALL = Path("C:/Users/emir-/Downloads/2026-10-07_biebrntionx/"
                "2026-10-07_biebrntionx_2107656533844828272_6lCwR1KHqzGZ6vJS.mp4")

CASES = [
    (FOOTBALL, [
        ("woman", "a woman", (0.0, 4.5)),
        ("players", "football players crying", (4.5, 9.3)),
        ("players2", "a crying man in a team jacket", (4.5, 9.3)),
    ]),
    (SPIKE / "testvideo_nolabels.mp4", [
        ("birthday", "birthday party", (20.0, 30.0)),
        ("sunny", "sunny warm weather", (0.0, 10.0)),
        ("tuesday", "quiet tuesday morning", (10.0, 20.0)),
        ("blue", "blue", (10.0, 20.0)),
    ]),
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
    dir_.mkdir(parents=True, exist_ok=True)
    span = ["-y", "-ss", str(start), "-t", str(win), "-i", str(video)]
    wav = dir_ / "clip.wav"
    run(span + ["-vn", "-ac", "1", "-ar", "16000", str(wav)])
    run(span + ["-vf", "fps=%d" % fps, "-q:v", "2", str(dir_ / "f%02d.jpg")])
    frames = sorted(str(p) for p in dir_.glob("f*.jpg"))
    return str(wav), frames


def embed(model, video, tag, start, win, fps):
    d = WORK / tag
    wav, frames = cut(d, start, win, fps, video)
    for payload in ({"image": frames, "audio": wav}, {"image": frames}):
        try:
            v = np.asarray(model.encode(payload), dtype=np.float32).reshape(-1)
            return v / (np.linalg.norm(v) + 1e-9)
        except Exception:
            continue
    raise RuntimeError("no payload worked for %s" % tag)


def grid(duration, win, stride, lo=0.0, hi=None):
    hi = duration if hi is None else min(hi, duration)
    out, s = [], lo
    while s + win <= hi + 1e-6:
        out.append(s)
        s += stride
    if not out:
        out = [max(0.0, min(lo, duration - win))]
    return out


def model_vec(model, text):
    q = np.asarray(model.encode(text, prompt_name="SearchQuery"),
                   dtype=np.float32).reshape(-1)
    return q / (np.linalg.norm(q) + 1e-9)


def overlaps(span, block):
    return span[0] < block[1] and span[1] > block[0]


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16})

    for video, queries in CASES:
        duration = probe_duration(video)
        tag = video.stem[:12]
        print("\n########## %s  (%.1f s) ##########" % (video.name, duration))

        coarse_starts = grid(duration, COARSE_W, COARSE_S)
        coarse = [embed(model, video, "cf_%s_coarse_%d" % (tag, i), s,
                        COARSE_W, FPS) for i, s in enumerate(coarse_starts)]
        coarse = np.stack(coarse)
        spans = [(s, min(s + COARSE_W, duration)) for s in coarse_starts]

        for label, text, block in queries:
            q = model_vec(model, text)
            cs = coarse @ q
            top = int(np.argmax(cs))
            cspan = spans[top]
            print("\n-- %-9s \"%s\"  truth %s" % (label, text, block))
            print("   coarse top %.3f  %5.1f-%5.1f  %s"
                  % (cs[top], cspan[0], cspan[1],
                     "OK" if overlaps(cspan, block) else "MISS"))

            lo = max(0.0, cspan[0] - FINE_S)
            fine_starts = grid(duration, FINE_W, FINE_S, lo=lo, hi=cspan[1])
            fine = [embed(model, video, "cf_%s_%s_%d" % (tag, label, i), s,
                          FINE_W, FINE_FPS) for i, s in enumerate(fine_starts)]
            fine = np.stack(fine)
            fs = fine @ q
            ftop = int(np.argmax(fs))
            fspan = (fine_starts[ftop], min(fine_starts[ftop] + FINE_W, duration))
            print("   fine   top %.3f  %5.1f-%5.1f  %s  (inside coarse %s)"
                  % (fs[ftop], fspan[0], fspan[1],
                     "OK" if overlaps(fspan, block) else "MISS",
                     "yes" if fspan[0] >= cspan[0] - 1e-6 and
                             fspan[1] <= cspan[1] + 1e-6 else "no"))
            order = np.argsort(-fs)[:3]
            print("   fine 3: " + " | ".join(
                "%5.1f-%-5.1f %.3f" % (fine_starts[i],
                                       min(fine_starts[i] + FINE_W, duration), fs[i])
                for i in order))


if __name__ == "__main__":
    main()
