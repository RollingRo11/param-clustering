"""Read back a newly created W&B validation run using the public API."""
import argparse
import json
from pathlib import Path
import time

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads((args.output / '.wandb' / 'validate_wandb.json').read_text())
    for attempt in range(12):
        api = wandb.Api(timeout=30)
        path = saved['url'].split('wandb.ai/', 1)[1].replace('/runs/', '/')
        run = api.run(path)
        history = list(run.scan_history())
        files = {f.name for f in run.files()}
        artifacts = list(run.logged_artifacts())
        try:
            assert run.project == 'param-autoenc'
            assert run.state == 'finished', run.state
            assert any(row.get('train/layer0.mlp_in/loss') == .5 for row in history)
            assert any('plots/validation.png' in row for row in history)
            # W&B custom charts store their table in history with a _table
            # suffix; the chart itself is configured separately by the SDK.
            assert any('curves/validation/kl_target_table' in row for row in history)
            console_name = next(name for name in files if name == 'output.log' or name.startswith('logs/output'))
            directory = args.output / 'remote_verification'
            console = run.file(console_name).download(root=str(directory), replace=True)
            text = Path(console.name).read_text()
            assert 'W&B stdout capture verification' in text
            assert 'W&B stderr capture verification' in text
            outputs = [a for a in artifacts if a.type == 'experiment-output']
            assert outputs
            names = set(outputs[0].manifest.entries)
            assert any(name.endswith('/report.md') for name in names)
            assert not any('old' in name for name in names)
        except (AssertionError, StopIteration):
            if attempt == 11:
                raise
            time.sleep(5)
            continue
        result = dict(url=saved['url'], project=run.project, state=run.state, history_rows=len(history),
                      console_stdout_verified=True, console_stderr_verified=True,
                      metrics_verified=True, image_verified=True, curve_verified=True,
                      output_artifact_files=sorted(names), historical_files_uploaded=False)
        (args.output / 'wandb_verified.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'WANDB_VERIFIED': result}), flush=True)
        break


if __name__ == '__main__':
    main()
