"""Bar chart of raw per-component attribution a_c(x) for single tokens, 1B model.
y = a_c(x) (squared IG inner, raw magnitude), x = component index 1..512."""

import sys

sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")

import torch

import geo1b  # noqa: F401 — applies the 1B patches to geo67
import geo67
from geo67 import GatedRunner

import nano_param_decomp.pile_4L as p4l

device = "cuda"
target = geo67.load_target(device)
bk = torch.load("/dev/shm/geo1b/run1/banks_prop1b.pt", weights_only=True,
                map_location="cpu")
run = GatedRunner(target, bk, device)
loader = p4l.make_loader(2, 256, 0, 1, "train", 777)
idx = next(loader).to(device)
attr, _ = run.attribution(idx, 2)          # [B, T, C], raw squared inners

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

picks = [(0, 64), (0, 128), (1, 192)]      # three representative tokens
fig, axes = plt.subplots(len(picks), 1, figsize=(11, 8), sharex=True)
for ax, (b, t) in zip(axes, picks):
    a = attr[b, t].float().cpu().numpy()
    frac = (a > 0.02 * a.max()).sum()
    ctx = tok.decode(idx[b, max(0, t - 8):t + 1].tolist()).replace("\n", " ")
    ax.bar(range(1, 513), a, width=1.0, color="steelblue")
    ax.set_ylabel("a_c(x)")
    ax.set_title(f"token {t}: …{ctx!r} — {frac}/512 components above gate "
                 f"threshold (2% of max)", fontsize=9, loc="left")
axes[-1].set_xlabel("component index (1..512)")
fig.suptitle("Raw per-component attribution a_c(x), Llama-3.2-1B, C=512 "
             "(linear scale, no normalization)", fontsize=11)
fig.tight_layout()
out = "/workspace/circuit-decomp/geo-attribution/out/full1b/ac_bar_1b.png"
fig.savefig(out, dpi=150)
print("saved", out)
for b, t in picks:
    a = attr[b, t].float().cpu()
    print(f"token ({b},{t}): max {a.max():.3e}, median {a.median():.3e}, "
          f">2%max: {(a > 0.02 * a.max()).sum().item()}, "
          f">10%max: {(a > 0.10 * a.max()).sum().item()}")
