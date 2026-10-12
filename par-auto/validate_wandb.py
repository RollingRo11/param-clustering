"""Small real logging payload for the online W&B integration test."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import parauto_tracking as tracking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tracking.start_cli(args)
    args.output.mkdir(parents=True, exist_ok=True)
    print('W&B stdout capture verification', flush=True)
    print('W&B stderr capture verification', file=sys.stderr, flush=True)
    for step in (0, 1):
        for matrix in ('layer0.mlp_in', 'layer0.mlp_out'):
            tracking.log_train(dict(step=step, loss=1/(step+1), total_loss=2/(step+1)), matrix=matrix)
            print(json.dumps(dict(step=step, matrix=matrix, loss=1/(step+1), total_loss=2/(step+1))), flush=True)
    result = dict(validation=True, config=dict(purpose='new logging integration smoke test'),
                  methods={'validation': dict(kept=[2, 1, 0], metrics={'kl_target': dict(mean=[0., .01, .1])})})
    (args.output / 'result.json').write_text(json.dumps(result))
    (args.output / 'report.md').write_text('# W&B logging validation\n\nNew integration test only; no historical run import.\n')
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [.1, .01, 0])
    ax.set(xlabel='Components kept', ylabel='Synthetic validation error', title='W&B integration check')
    fig.savefig(args.output / 'validation.png')
    plt.close(fig)
    print('WANDB_PAYLOAD_COMPLETE', flush=True)


if __name__ == '__main__':
    tracking.run_main(main)
