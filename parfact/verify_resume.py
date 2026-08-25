from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch", "numpy"]))

@function(image=IMG, gpu="H100", pool="ondemand-h100", cpu=4, memory="16Gi",
          disks=[DurableDisk(name="spd-c600", size="10Gi", mount_path="/data")],
          timeout=600, retries=0)
def cmp():
    import torch, os
    out = {}
    for tag in ("vA", "vC"):
        p = f"/data/runs/{tag}/spd_state.pt"
        out[tag] = os.path.exists(p)
    if not all(out.values()):
        return {"exists": out}
    a = torch.load("/data/runs/vA/spd_state.pt", weights_only=True)
    b = torch.load("/data/runs/vC/spd_state.pt", weights_only=True)
    worst = 0.0
    for n in a["wrappers"]:
        for k in ("V", "U"):
            d = (a["wrappers"][n][k] - b["wrappers"][n][k]).abs().max().item()
            worst = max(worst, d)
    gdiff = max((a["gates"][k] - b["gates"][k]).abs().max().item()
                for k in a["gates"])
    return {"exists": out, "max_abs_diff_VU": worst, "max_abs_diff_gates": gdiff}

if __name__ == "__main__":
    print(cmp.remote())
