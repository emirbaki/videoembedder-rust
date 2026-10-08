"""Compare the Rust GGUF embedding against the Python safetensors embedding.

Run the Rust binary first. It writes spike/parity_rust.txt.
"""
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path("D:/MLProjelerim/videoembedder-rust")
TEXT = "task: search result | query: birthday party"


def main():
    model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16})
    v = np.asarray(model.encode(TEXT, normalize_embeddings=True), dtype=np.float32)
    r = np.loadtxt(ROOT / "spike/parity_rust.txt", dtype=np.float32)

    print("python dim %s   rust dim %s" % (v.shape, r.shape))
    print("cosine        %.6f" % float(np.dot(v, r)))
    print("max abs diff  %.6f" % float(np.max(np.abs(v - r))))
    print("mean abs diff %.6f" % float(np.mean(np.abs(v - r))))
    print("python norm   %.6f" % float(np.linalg.norm(v)))
    print("rust norm     %.6f" % float(np.linalg.norm(r)))


if __name__ == "__main__":
    main()
