from beam import Image, function
from beta9.type import DurableDisk

@function(image=Image(python_version="python3.12"), cpu=1, memory="2Gi",
          disks=[DurableDisk(name="cofac67", size="50Gi",
                             mount_path="/data")], timeout=300, retries=0)
def get():
    return open("/data/cofac67/klkeep_vpd.json").read()

if __name__ == "__main__":
    import pathlib
    t = get.remote()
    pathlib.Path("out/klkeep_vpd.json").write_text(t)
    print("SAVED", len(t))
