import base64
from beam import Image, function
from beta9.type import DurableDisk

@function(image=Image(python_version="python3.12"), cpu=1, memory="2Gi",
          disks=[DurableDisk(name="parfact-disk", size="10Gi", mount_path="/data")],
          timeout=300, retries=0)
def get():
    return base64.b64encode(open("/data/induction_model.pt", "rb").read()).decode()

if __name__ == "__main__":
    blob = get.remote()
    open("induction_model_100k.pt", "wb").write(base64.b64decode(blob))
    print("fetched induction_model_100k.pt")
