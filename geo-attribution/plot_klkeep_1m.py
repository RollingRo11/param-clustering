"""Percent-kept KLKeep: 1M-event cofac (C=4096) vs VPD, shared 96 holdout
events. Reads the cluster-produced JSONs from out/; rerun after ep48 eval
lands to add its line."""
import json
import pathlib
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).parent / "out"
FIG = pathlib.Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

cofac = json.load(open(OUT / "klkeep_big_ep12.json"))
vpd = json.load(open(OUT / "klkeep_vpd_big.json"))
C = 4096
FLOOR = 1e-7

def series(d, xk):
    xs = sorted(d, key=float)
    x = [xk(k) for k in xs]
    kl = [max(d[k]["kl_mean"], FLOOR) for k in xs]
    rnd = [max(d[k]["kl_rand_mean"], FLOOR) for k in xs]
    return x, kl, rnd

cx, ckl, crnd = series(cofac, lambda k: 100 * int(k) / C)
vx, vkl, vrnd = series(vpd, float)

ep48_path = OUT / "klkeep_big_ep48.json"
ep48 = json.load(open(ep48_path)) if ep48_path.exists() else None

plt.figure(figsize=(6.4, 4.4))
plt.plot(cx, ckl, "o-", color="tab:blue", label="co-fac (1M events, C=4096)")
plt.plot(cx, crnd, "o--", color="tab:blue", alpha=0.35,
         label="co-fac random-k")
plt.plot(vx, vkl, "s-", color="tab:red", label="VPD (CI order)")
plt.plot(vx, vrnd, "s--", color="tab:red", alpha=0.35, label="VPD random-k")
if ep48 is not None:
    ex, ekl, _ = series(ep48, lambda k: 100 * int(k) / C)
    plt.plot(ex, ekl, "^-", color="tab:green", label="co-fac (48 epochs)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("% of components kept")
plt.ylabel("KL(kept ‖ full) nats")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "klkeep67_1m_vs_vpd.png", dpi=160)
print("saved", FIG / "klkeep67_1m_vs_vpd.png")
