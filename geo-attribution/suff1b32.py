"""Per-token sufficiency + set-cover refinement on the flagship 1B banks."""
import json, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM

dev = "cuda:0"; torch.cuda.set_device(0)
B = torch.load("/dev/shm/geo1b/fl32_streamC8192/banks_flagship.pt",
               map_location="cpu", weights_only=False)
MODS, C, K = B["modules"], int(B["C"]), 2
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B",
    revision="9535bd9b1d1dea6acafbdc4813b728796aeb28da",
    torch_dtype=torch.float32).to(dev).eval()
def sub(m): return model.get_submodule(m[3:] if m.startswith("hf.") else m)
sidx = {m: B["sidx"][m].to(dev) for m in MODS}
swgt = {m: B["swgt"][m].to(dev) for m in MODS}
W0 = {m: sub(m).weight.detach().clone() for m in MODS}
Wt = {m: sub(m).weight for m in MODS}
arr = np.memmap("/dev/shm/geo1b/pile_llama_u32.bin", dtype=np.uint32, mode="r")
tail = torch.from_numpy(np.asarray(arr[-40*512:], dtype=np.int64)).view(40, 512).to(dev)
ref_ids = torch.from_numpy(np.asarray(arr[-90*512:-45*512], dtype=np.int64)).view(45, 512).to(dev)
g = torch.Generator().manual_seed(12345)
samp = [(int(torch.randint(0, 40, (1,), generator=g)),
         int(torch.randint(64, 510, (1,), generator=g))) for _ in range(24)]
gr = torch.Generator().manual_seed(555)
refs = [(int(torch.randint(0, 45, (1,), generator=gr)),
         int(torch.randint(64, 510, (1,), generator=gr))) for _ in range(128)]

def restore():
    with torch.no_grad():
        for m in MODS: Wt[m].copy_(W0[m])

def tok_ce(ids, b, t):
    with torch.no_grad():
        lg = model(ids[b:b+1]).logits[0, t].float()
        return float(F.cross_entropy(lg[None], ids[b, t+1][None]))

def attr(ids, b, t):
    A = {m: torch.zeros_like(W0[m]) for m in MODS}
    for st in range(K):
        a = (st+1)/K
        with torch.no_grad():
            for m in MODS: Wt[m].copy_(W0[m]*a)
        pre, post, hs = {}, {}, []
        for m in MODS:
            def hk(mm, i, o, _m=m):
                pre[_m] = i[0]; o.retain_grad(); post[_m] = o
            hs.append(sub(m).register_forward_hook(hk))
        lg = model(ids[b:b+1]).logits
        for h in hs: h.remove()
        rw = lg[0, t, ids[b, t+1]].float()
        gs = torch.autograd.grad(rw, [post[m] for m in MODS])
        for m, gg in zip(MODS, gs):
            A[m] += (gg[0].float().t() @ pre[m][0].float())/K
        del pre, post, gs, lg
    restore()
    return {m: A[m]*W0[m] for m in MODS}

def scores(AW, sw):
    v = torch.zeros(C, device=dev, dtype=torch.float64)
    with torch.no_grad():
        for m in MODS:
            v += torch.bincount(sidx[m].reshape(-1).int(),
                weights=(sw[m].float()*AW[m][None]).reshape(-1).double(), minlength=C)
    return v

restore(); base = float(np.mean([tok_ce(tail, b, t) for b, t in samp]))
print(f"base CE {base:.4f}", flush=True)
# refine
t0 = time.perf_counter()
acc = {m: torch.zeros_like(swgt[m], dtype=torch.float32) for m in MODS}
for b, t in refs:
    AW = attr(ref_ids, b, t); sc = scores(AW, swgt)
    o = torch.argsort(sc, descending=True)
    rk = torch.empty(C, device=dev); rk[o] = torch.arange(C, device=dev, dtype=torch.float32)
    gn = 1.0/(rk+16)
    with torch.no_grad():
        for m in MODS: acc[m] += AW[m].abs()[None]*gn[sidx[m].int()]
    del AW
swr = {}
with torch.no_grad():
    for m in MODS:
        w = swgt[m].float()*acc[m]; tot = w.sum(0, keepdim=True)
        swr[m] = torch.where(tot <= 0, swgt[m].float(), w/tot.clamp_min(1e-30)).half()
del acc
print(f"refined ({time.perf_counter()-t0:.0f}s)", flush=True)
KEEP = [0, 64, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 28672, C]
out = {"C": C, "base": round(base, 4), "keep": KEEP, "n": len(samp)}
for name, sw in (("original", swgt), ("refined", swr)):
    cur = np.zeros((len(samp), len(KEEP)))
    for j, (b, t) in enumerate(samp):
        AW = attr(tail, b, t); sc = scores(AW, sw); del AW
        o = torch.argsort(sc, descending=True)
        rk = torch.empty(C, dtype=torch.int32, device=dev)
        rk[o] = torch.arange(C, dtype=torch.int32, device=dev)
        with torch.no_grad():
            R = {m: rk[sidx[m].int()] for m in MODS}
            for ki, kk in enumerate(KEEP):
                for m in MODS:
                    Wt[m].copy_(W0[m]*(sw[m].float()*(R[m] < kk)).sum(0, dtype=torch.float32))
                cur[j, ki] = tok_ce(tail, b, t)
            del R
        restore()
    mu = cur.mean(0)
    thr = next((k for k, v in zip(KEEP, mu) if v-base <= 0.25), C)
    out[name] = {"ce": [round(float(v), 4) for v in mu], "k": thr,
                 "rt": round(float(mu[-1]-base), 5)}
    print(f"{name}: k={thr}/{C} ({100*thr/C:.1f}%) CE@256 {mu[KEEP.index(256)]:.2f} rt {out[name]['rt']:+.1e}", flush=True)
open("out/suff1b_32k.json", "w").write(json.dumps(out, indent=1))
