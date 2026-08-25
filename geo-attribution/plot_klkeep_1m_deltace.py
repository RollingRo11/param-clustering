"""klkeep67_1m_linear variant: y axis is the raw delta CE (nats, KL to the
full model) instead of % CE lost. x = % components deleted."""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).parent / "out"
FIG = pathlib.Path(__file__).parent / "figures"
cofac = json.load(open(OUT / "klkeep_big_ep12.json"))
vpd = json.load(open(OUT / "klkeep_vpd_big.json"))
C = 4096

def series(d, xk):
    xs = sorted(d, key=float)
    x = [xk(k) for k in xs]
    kl = [d[k]["kl_mean"] for k in xs]
    rnd = [d[k]["kl_rand_mean"] for k in xs]
    return x, kl, rnd

cx, ckl, crnd = series(cofac, lambda k: 100 - 100 * int(k) / C)
vx, vkl, vrnd = series(vpd, lambda p: 100 - float(p))

plt.figure(figsize=(6.4, 4.4))
plt.plot(cx, ckl, "o-", color="tab:blue", label="co-fac")
plt.plot(cx, crnd, "o--", color="tab:blue", alpha=0.35)
plt.plot(vx, vkl, "s-", color="tab:red", label="VPD")
plt.plot(vx, vrnd, "s--", color="tab:red", alpha=0.35)
plt.xlim(0, 100)
plt.ylim(0, 68)
plt.xlabel("% of components deleted")
plt.ylabel("ΔCE (nats)")
plt.legend(loc="upper left")
ax = plt.gca().inset_axes([0.08, 0.35, 0.6, 0.45])
ax.plot(cx, ckl, "o-", color="tab:blue", ms=4)
ax.plot(vx, vkl, "s-", color="tab:red", ms=4)
ax.set_xlim(90, 100)
ax.set_ylim(0, 5)
ax.set_title("zoom: solid lines", fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "klkeep67_1m_deltace.png", dpi=160)
print("saved")
