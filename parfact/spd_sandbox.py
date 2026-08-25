"""Run the C=600 SPD decomposition in a detached Beam Sandbox.

`.remote()` is unusable for this job: it ties the container's lifetime to the
local client's gRPC stream, and when that stream broke ~8 minutes in the task
was CANCELLED mid-training. A Sandbox lives independently of this process, so
the run survives disconnects.

    python spd_sandbox.py start     # launch, prints the sandbox id
    python spd_sandbox.py status    # progress (reconnects by id)
    python spd_sandbox.py fetch     # download artifacts once finished
    python spd_sandbox.py stop      # terminate the sandbox
"""
import sys
from pathlib import Path

from beam import Image, Sandbox
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch", "matplotlib", "numpy"]))
DISK = [DurableDisk(name="spd-c600", size="10Gi", mount_path="/data")]
ID_FILE = Path("spd_sandbox_id.txt")
LOG = "/data/spd_C600.log"
OUT = "/data/runs/spd_C600"

# spd_toy -> induction_model, vpd_toy -> prev_method -> atoms -> ...
# The import chain is deeper than it looks, so ship every module.
FILES = sorted(f.name for f in Path(".").glob("*.py")) + \
    ["induction_model_100k.pt"]

CMD = (f"cd /workspace && mkdir -p {OUT} && "
       # -u: stdout redirected to a file is block-buffered, so progress
       # would otherwise sit invisible in a 4KB buffer for many minutes.
       f"nohup python3 -u spd_toy.py --c_per_module 100 --steps 100000 "
       f"--ckpt induction_model_100k.pt --out {OUT} --world 1 "
       f"> {LOG} 2>&1 & echo started")


# 3h ceiling rather than -1: the run needs ~2h, and a sandbox with no timeout
# that gets forgotten quietly bills a GPU forever.
TTL = 10800


def start():
    sb = Sandbox(name="spd-c600", image=IMG, gpu="A10G", cpu=4,
                 memory="16Gi", disks=DISK, keep_warm_seconds=TTL).create()
    # The constructor's keep_warm_seconds did NOT stick -- the first sandbox
    # died on the default 600s despite asking for 10800. Set it explicitly,
    # and refresh on every attach below.
    sb.update_ttl(TTL)
    launch(sb)
    sid = sb.sandbox_id()          # a method, not a property
    ID_FILE.write_text(sid)
    print(f"sandbox {sid}  (id saved to {ID_FILE})")


def launch(sb):
    """(Re)upload sources and start training inside an existing sandbox."""
    for f in FILES:
        sb.fs.upload_file(f, f"/workspace/{f}")
    p = sb.process.exec("bash", "-lc", CMD)
    p.wait()
    print("".join(p.logs))
    sid = sb.sandbox_id()
    ID_FILE.write_text(sid)
    print(f"sandbox {sid}  (id saved to {ID_FILE})")


def restart():
    launch(_attach())
    print("relaunched")


def _attach():
    sb = Sandbox().connect(ID_FILE.read_text().strip())
    sb.update_ttl(TTL)      # refresh the countdown on every interaction
    return sb


def status():
    sb = _attach()
    p = sb.process.exec("bash", "-lc",
                        f"tail -5 {LOG}; echo ---; "
                        f"ls -la {OUT} 2>/dev/null | tail -5; echo ---; "
                        f"pgrep -f spd_toy.py >/dev/null && echo RUNNING || echo NOT_RUNNING")
    p.wait()
    print("".join(p.logs))


def fetch():
    sb = _attach()
    Path("out/spd_C600").mkdir(parents=True, exist_ok=True)
    p = sb.process.exec("bash", "-lc", f"ls {OUT}")
    p.wait()
    for name in "".join(p.logs).split():
        sb.fs.download_file(f"{OUT}/{name}", f"out/spd_C600/{name}")
        print("downloaded", name)


if __name__ == "__main__":
    {"start": start, "status": status, "fetch": fetch, "restart": restart,
     "stop": lambda: (_attach().terminate(), print("terminated"))}[sys.argv[1]]()
