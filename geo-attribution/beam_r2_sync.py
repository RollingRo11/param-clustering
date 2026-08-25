"""Sync ALL Beam durable disks to R2 in one job (module-level handlers).

    python beam_r2_sync.py probe
    python beam_r2_sync.py sync
    python beam_r2_sync.py verify
"""
import json
import sys
from beam import Image, function
from beta9.type import DurableDisk

BUCKET = "param-clustering"
IMG = Image(python_version="python3.12").add_python_packages(["boto3"])
NAMES = ["cofac67", "spd-c600", "spd-c600-paper", "spd-c600-s1",
         "spd-c600-concat", "spd-c600-cc", "cofact-c600", "parfact-disk"]
DISKS = [DurableDisk(name=d, size="50Gi" if d == "cofac67" else "10Gi",
                     mount_path=f"/src/{d}") for d in NAMES]
SECRETS = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT"]


def _client():
    import os, boto3
    return boto3.client(
        "s3", endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto")


def _remote_index(s3, prefix=""):
    have = {}
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            have[o["Key"]] = o["Size"]
    return have


@function(image=IMG, cpu=2, memory="4Gi", timeout=600, retries=0,
          secrets=SECRETS)
def probe():
    _client().head_bucket(Bucket=BUCKET)
    return {"bucket_ok": True}


@function(image=IMG, cpu=4, memory="8Gi", timeout=3600 * 3, retries=0,
          secrets=SECRETS, disks=DISKS)
def sync():
    import os
    s3 = _client()
    have = _remote_index(s3)
    out = {}
    for d in NAMES:
        up = skipped = bytes_up = 0
        for root, _, files in os.walk(f"/src/{d}"):
            for f in files:
                path = os.path.join(root, f)
                key = f"{d}/" + os.path.relpath(path, f"/src/{d}")
                size = os.path.getsize(path)
                if have.get(key) == size:
                    skipped += 1
                    continue
                s3.upload_file(path, BUCKET, key)
                up += 1
                bytes_up += size
        out[d] = {"uploaded": up, "skipped": skipped, "bytes": bytes_up}
        print(json.dumps({d: out[d]}), flush=True)
    return out


@function(image=IMG, cpu=2, memory="4Gi", timeout=3600, retries=0,
          secrets=SECRETS, disks=DISKS)
def verify():
    import os
    s3 = _client()
    remote = _remote_index(s3)
    out = {}
    for d in NAMES:
        local, mismatch = {}, []
        for root, _, files in os.walk(f"/src/{d}"):
            for f in files:
                path = os.path.join(root, f)
                key = f"{d}/" + os.path.relpath(path, f"/src/{d}")
                local[key] = os.path.getsize(path)
                if remote.get(key) != local[key]:
                    mismatch.append(key)
        out[d] = {"local_files": len(local),
                  "local_bytes": sum(local.values()),
                  "mismatched": mismatch[:8]}
    return out


if __name__ == "__main__":
    fn = {"probe": probe, "sync": sync, "verify": verify}[sys.argv[1]]
    r = fn.remote()
    print(json.dumps(r, indent=1) if r else "FAILED")
