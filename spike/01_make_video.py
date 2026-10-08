"""Make spike/testvideo.mp4: 30 s, three 10 s blocks, each with speech.

Block 0: "SUNNY DAY"        speech: the weather is sunny and warm today
Block 1: "TUESDAY MORNING"  speech: it is a quiet tuesday morning
Block 2: "BIRTHDAY PARTY"   speech: we are here to celebrate a birthday party
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("D:/MLProjelerim/videoembedder-rust")
FFMPEG = ROOT / "tools/dl/ff/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
WORK = ROOT / "spike/work"
WORK.mkdir(parents=True, exist_ok=True)

SEGMENTS = [
    ("darkorange", "SUNNY DAY", "The weather is sunny and warm today", "seg1"),
    ("navy", "TUESDAY MORNING", "It is a quiet tuesday morning", "seg2"),
    ("darkgreen", "BIRTHDAY PARTY", "We are here to celebrate a birthday party", "seg3"),
]


def speak(text, wav):
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.SetOutputToWaveFile('" + str(wav) + "'); "
        "$s.Speak('" + text + "')"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script],
                   check=True, capture_output=True)


def find_font():
    for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        src = Path("C:/Windows/Fonts") / name
        if src.exists():
            shutil.copy(src, WORK / "font.ttf")
            return True
    return False


def make_segment(color, tag, vf):
    base = [str(FFMPEG), "-y", "-f", "lavfi", "-i",
            "color=c=" + color + ":s=480x270:r=30", "-i", tag + ".wav", "-t", "10"]
    tail = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "16000", "-ac", "1", tag + ".mp4"]
    cmd = base + (["-vf", vf] if vf else []) + tail
    try:
        subprocess.run(cmd, cwd=WORK, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(base + tail, cwd=WORK, check=True, capture_output=True)
        print("drawtext failed on", tag, "so that block has no label")


def main():
    nolabels = "--nolabels" in sys.argv
    name = "testvideo_nolabels.mp4" if nolabels else "testvideo.mp4"
    suffix = "_nolabels" if nolabels else ""
    has_font = find_font() and not nolabels
    for color, label, text, tag in SEGMENTS:
        speak(text, WORK / (tag + suffix + ".wav"))
        if not nolabels:
            (WORK / (tag + ".txt")).write_text(label, encoding="ascii")
        vf = None
        if has_font:
            vf = ("drawtext=fontfile=font.ttf:textfile=" + tag + ".txt"
                  ":fontcolor=white:fontsize=30:x=20:y=20")
        make_segment(color, tag + suffix, vf)
    (WORK / ("list" + suffix + ".txt")).write_text(
        "".join("file '" + s[3] + suffix + ".mp4'\n" for s in SEGMENTS),
        encoding="ascii")
    subprocess.run([str(FFMPEG), "-y", "-f", "concat", "-safe", "0", "-i",
                    "list" + suffix + ".txt", "-c", "copy", "../" + name],
                   cwd=WORK, check=True, capture_output=True)
    print("wrote", ROOT / "spike" / name, "| labels:", not nolabels)


if __name__ == "__main__":
    main()
