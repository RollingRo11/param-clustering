from beam import Image, function
from beta9.type import DurableDisk

@function(image=Image(python_version="python3.12"), cpu=1, memory="2Gi",
          disks=[DurableDisk(name="spd-c600", size="10Gi", mount_path="/spd")],
          timeout=300, retries=0)
def get():
    return open("/spd/runs/spd_C600/analysis.json").read()

if __name__ == "__main__":
    txt = get.remote()
    import pathlib
    pathlib.Path("out/spd_analysis_C600.json").write_text(txt)
    print("SAVED", len(txt), "bytes")
