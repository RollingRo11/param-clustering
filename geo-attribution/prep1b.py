"""One-time prep for the Llama-3.2-1B port: download model + tokenizer to
/dev/shm/hf and pretokenize ~60M tokens of pile-uncopyrighted into a flat
uint32 bin (the 1B loader reads fixed-length rows from it)."""

import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/dev/shm/hf")

MODEL_ID = "unsloth/Llama-3.2-1B"
SHM = Path("/dev/shm/geo1b")
OUT = SHM / "pile_llama_u32.bin"
TARGET_TOKENS = 60_000_000


def main():
    SHM.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    p = snapshot_download(MODEL_ID)
    print(f"model snapshot at {p}", flush=True)
    if OUT.exists() and OUT.stat().st_size >= TARGET_TOKENS * 4:
        print(f"bin already complete ({OUT.stat().st_size} bytes)", flush=True)
        return
    from transformers import AutoTokenizer
    import datasets
    import array
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = datasets.load_dataset("monology/pile-uncopyrighted", split="train",
                               streaming=True)
    arr = array.array("I")
    buf, t0 = [], time.time()
    for ex in ds:
        buf.append(ex["text"])
        if len(buf) == 256:
            for ids in tok(buf)["input_ids"]:
                arr.extend(ids)
            buf = []
            if len(arr) >= TARGET_TOKENS:
                break
            if len(arr) % 5_000_000 < 60_000:
                print(f"{len(arr)/1e6:.0f}M tokens ({time.time()-t0:.0f}s)",
                      flush=True)
    with open(OUT, "wb") as f:
        arr[:TARGET_TOKENS].tofile(f)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
