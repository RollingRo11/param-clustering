#!/usr/bin/env python3
"""Pull factorization.pt for one or more cofact runs out of R2 onto local disk.

The runs live on the Beam DurableDisk `cofact-c600`, which cannot be mounted
outside Beam. beam_r2_sync.py mirrors every disk into the `param-clustering`
bucket under the key prefix `<disk-name>/`, so on the Explorer cluster we pull
from R2 instead (egress works from compute nodes).

    python r2_pull_fact.py --dest data/component_mass \
        B_layer_K1200_C600_v2_idiv_usimplex_ssimplex \
        B_layer_K1200_C600_v2_idiv_usimplex

Needs R2_ENDPOINT / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY in the environment
(`source env.sh` on the cluster).
"""
import argparse
import os
from pathlib import Path

BUCKET = "param-clustering"
DISK = "cofact-c600"


def client():
    import boto3
    missing = [v for v in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID",
                           "R2_SECRET_ACCESS_KEY") if not os.environ.get(v)]
    if missing:
        raise SystemExit(f"missing env: {', '.join(missing)} -- source env.sh")
    return boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"],
                        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                        region_name="auto")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directory names under runs/")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--file", default="factorization.pt")
    args = ap.parse_args()

    s3 = client()
    for run in args.runs:
        key = f"{DISK}/runs/{run}/{args.file}"
        out = args.dest / run / args.file
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
        except Exception as e:
            # a run that was never synced (or never existed) should name itself
            raise SystemExit(f"not in R2: s3://{BUCKET}/{key}\n  {e}")
        if out.exists() and out.stat().st_size == size:
            print(f"have  {out}  ({size:,} B)", flush=True)
            continue
        print(f"pull  {key} -> {out}  ({size:,} B)", flush=True)
        s3.download_file(BUCKET, key, str(out))


if __name__ == "__main__":
    main()
