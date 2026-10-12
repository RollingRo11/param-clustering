"""Fetch the public paper decomposition without uploading anything to W&B."""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from parauto_target import atomic_json, fingerprint  # noqa: E402


def run_metadata(project, run_id):
    query = """query($project: String!, $run: String!) {
      project(name: $project, entityName: "goodfire") {
        run(name: $run) { name state config
          files(first: 100) { edges { node { name sizeBytes url } } }
        }
      }
    }"""
    request = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": query, "variables": {"project": project, "run": run_id}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if payload.get("errors"):
        raise RuntimeError("W&B metadata query failed")
    metadata = payload["data"]["project"]["run"]
    if metadata is None or metadata["state"] != "finished":
        raise RuntimeError("Published run is missing or unfinished")
    config = {key: value["value"] for key, value in json.loads(metadata["config"]).items()
              if isinstance(value, dict) and "value" in value and not key.startswith("_")}
    files = {entry["node"]["name"]: entry["node"] for entry in metadata["files"]["edges"]}
    return config, files


def download_file(info, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".download")
    with urllib.request.urlopen(info["url"], timeout=60) as response, temporary.open("wb") as output:
        while block := response.read(1048576):
            output.write(block)
    if temporary.stat().st_size != info["sizeBytes"]:
        raise RuntimeError(f"Incomplete download: {destination.name}")
    temporary.replace(destination)


import parauto_tracking as tracking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tracking.start_cli(args)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    cfg, files = run_metadata("spd-resid-mlp", "wau744ht")
    if cfg["pretrained_model_path"] != "wandb:spd-train-resid-mlp/runs/i1fk7jmb":
        raise ValueError("Released decomposition target mismatch")
    if cfg["steps"] != 50000:
        raise ValueError("Expected the 50k-step paper decomposition")
    atomic_json(cfg, root / "run_config.json")
    print(json.dumps({"config": cfg, "files": {k: v["sizeBytes"] for k, v in files.items()}}), flush=True)
    names = ["model_50000.pth", "final_config.yaml"]
    manifest = {}
    for name in names:
        if name not in files:
            raise ValueError(f"Missing released file: {name}")
        destination = root / name
        download_file(files[name], destination)
        manifest[name] = {"bytes": destination.stat().st_size, "sha256": fingerprint(destination)}
    atomic_json({"run": "goodfire/spd-resid-mlp/wau744ht", "files": manifest}, root / "manifest.json")
    print("SPD_ASSETS_READY", flush=True)


if __name__ == "__main__":
    tracking.run_main(main)
