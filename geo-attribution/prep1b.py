"""Download the 1B model and stream-tokenize Pile into a uint32 file."""

import argparse
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

MODEL_ID = os.environ.get("GEO_MODEL_ID", "unsloth/Llama-3.2-1B")
DEFAULT_MODEL_REVISION = "9535bd9b1d1dea6acafbdc4813b728796aeb28da"
MODEL_REVISION = os.environ.get("GEO_MODEL_REVISION", DEFAULT_MODEL_REVISION)
SHM = Path(os.environ.get("GEO_ATTRIBUTION_ARTIFACT_ROOT", "/dev/shm/geo1b"))
OUT = Path(os.environ.get("GEO_ATTRIBUTION_DATA_PATH",
                          str(SHM / "pile_llama_u32.bin")))
DEFAULT_TARGET_TOKENS = int(os.environ.get(
    "GEO_ATTRIBUTION_TARGET_TOKENS", "60000000"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    ap.add_argument("--text_batch", type=int, default=256)
    args = ap.parse_args()
    if args.target_tokens < 1 or args.text_batch < 1:
        ap.error("target_tokens and text_batch must be positive")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    p = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    print(f"model snapshot at {p}", flush=True)
    if OUT.exists() and OUT.stat().st_size >= args.target_tokens * 4:
        print(f"bin already complete ({OUT.stat().st_size} bytes)", flush=True)
        return
    from transformers import AutoTokenizer
    import datasets
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    ds = datasets.load_dataset("monology/pile-uncopyrighted", split="train",
                               streaming=True)
    partial = OUT.with_suffix(OUT.suffix + ".partial")
    written = 0
    buf, t0 = [], time.time()
    import numpy as np
    with open(partial, "wb") as output:
        for ex in ds:
            buf.append(ex["text"])
            if len(buf) < args.text_batch:
                continue
            encoded = tok(buf)["input_ids"]
            buf = []
            flat = np.fromiter((token for ids in encoded for token in ids),
                               dtype=np.uint32)
            take = min(flat.size, args.target_tokens - written)
            flat[:take].tofile(output)
            written += take
            if written % 5_000_000 < max(take, 1):
                print(f"{written/1e6:.0f}M tokens ({time.time()-t0:.0f}s)",
                      flush=True)
            if written == args.target_tokens:
                break
        output.flush()
        os.fsync(output.fileno())
    if written != args.target_tokens:
        raise RuntimeError(
            f"dataset ended after {written} tokens, wanted {args.target_tokens}")
    os.replace(partial, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
