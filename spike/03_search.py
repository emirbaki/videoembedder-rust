"""Search one video by text. Run after 02_embed_windows.py.

Usage: python 03_search.py "birthday party"
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path("D:/MLProjelerim/videoembedder-rust")
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "embeddings"
NPZ = ROOT / ("spike/" + PREFIX + ".npz")


def main():
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "birthday party"
    d = np.load(NPZ)
    vecs, starts, ends = d["vecs"], d["starts"], d["ends"]

    model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16})
    q = np.asarray(model.encode(query, prompt_name="SearchQuery"), dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)

    scores = vecs @ q
    order = np.argsort(-scores)

    print("query: %s" % query)
    print("windows: %d, dim %d" % (len(vecs), vecs.shape[1]))
    print()
    for i, idx in enumerate(order[:5], 1):
        bar = "#" * max(0, int(round(scores[idx] * 40)))
        print("  %d. %5.1f-%5.1f s   %.3f  %s"
              % (i, starts[idx], ends[idx], scores[idx], bar))


if __name__ == "__main__":
    main()
