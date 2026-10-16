"""Explicit W&B logging for par-auto training and evaluation code.

Metrics enter through log_train/log_metrics/log_result, never console parsing.
W&B's normal console capture is retained independently for human-readable logs.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = 'param-autoenc'
SMALL_SUFFIXES = {'.json', '.md', '.png', '.svg', '.txt', '.log', '.csv', '.yaml', '.yml'}
_run = None
_wandb = None
_roots = []
_before = {}
_started = None
_exit_code = 0
_matrix_names = []

def flat_scalars(value, prefix='', limit=2000):
    result = {}
    def visit(item, key):
        if len(result) >= limit:
            return
        if isinstance(item, dict):
            for name, child in item.items():
                visit(child, f'{key}/{name}' if key else str(name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            result[key] = item
    visit(value, prefix)
    return result


def output_roots(arguments):
    roots = []
    for i, value in enumerate(arguments):
        if value in ('--output', '--run', '--root') and i+1 < len(arguments):
            path = Path(arguments[i+1]).resolve()
            roots.append(path.parent if path.suffix in SMALL_SUFFIXES else path)
    # Report scripts accept their experiment directory positionally.
    if arguments and not arguments[0].startswith('-') and Path(arguments[0]).is_dir():
        roots.append(Path(arguments[0]).resolve())
    return list(dict.fromkeys(roots))


def candidate_files(roots):
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if any(part in ('wandb', '.wandb', '__pycache__', '.pytest_cache') for part in relative.parts):
                continue
            is_model = path.name in ('pieces.pt', 'checkpoint.pt') or (path.name.startswith('model_') and path.suffix == '.pth')
            if path.suffix in SMALL_SUFFIXES or is_model:
                yield root, path


def snapshot(roots):
    return {str(path): (path.stat().st_mtime_ns, path.stat().st_size) for _, path in candidate_files(roots)}


def publish_outputs(run, wandb, roots, before):
    artifact = wandb.Artifact(f'outputs-{run.id}', type='experiment-output')
    seen, count = set(), 0
    for index, root in enumerate(roots):
        for _, path in candidate_files([root]):
            if str(path) in seen:
                continue
            seen.add(str(path))
            stat = path.stat()
            if before.get(str(path)) == (stat.st_mtime_ns, stat.st_size):
                continue
            relative = path.relative_to(root).as_posix()
            key = f'output{index}/{relative}'
            artifact.add_file(str(path), name=key)
            count += 1
            if path.suffix == '.png':
                run.log({f'plots/{relative}': wandb.Image(str(path))})
            elif path.suffix == '.json' and stat.st_size < 20_000_000:
                try:
                    payload = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                # Curves keep their full arrays in the artifact. Numeric leaf
                # summaries (including selected budgets) remain searchable.
                run.summary.update(flat_scalars(payload, f'results/{relative}'))
                if isinstance(payload, dict):
                    if path.name == 'result.json':
                        settings = {key: payload[key] for key in ('config', 'training_config', 'sensor', 'precision',
                                    'target_sha256', 'components_per_matrix', 'steps_per_matrix') if key in payload}
                        if settings:
                            run.config.update(settings, allow_val_change=True)
                    for method, curve in payload.get('methods', {}).items():
                        if not isinstance(curve, dict) or 'kept' not in curve or 'metrics' not in curve:
                            continue
                        for metric in ('kl_target', 'mse_target'):
                            if metric not in curve['metrics']:
                                continue
                            table = wandb.Table(columns=['components_kept', 'mean_error'],
                                                data=list(map(list, zip(curve['kept'], curve['metrics'][metric]['mean']))))
                            run.log({f'curves/{method}/{metric}': wandb.plot.line(
                                table, 'components_kept', 'mean_error', title=f'{method}: {metric}')})
    if count:
        run.log_artifact(artifact)
    run.summary['output_file_count'] = count


def start(kind, output, config, *, resume=False, extra_roots=()):
    global _run, _wandb, _roots, _before, _started, _exit_code
    if _run is not None:
        raise RuntimeError('A W&B run is already active')
    import wandb
    _wandb = wandb
    base = Path(output).resolve()
    base.mkdir(parents=True, exist_ok=True)
    _roots = list(dict.fromkeys([base, *[Path(p).resolve() for p in extra_roots]]))
    _before = snapshot(_roots)
    directory = base / '.wandb'
    directory.mkdir(exist_ok=True)
    state_path = directory / f'{kind}.json'
    previous = json.loads(state_path.read_text()) if resume and state_path.exists() else None
    job = os.environ.get('SLURM_JOB_ID', 'local')
    task = os.environ.get('SLURM_ARRAY_TASK_ID', '')
    # JSON encoding converts pathlib/namespace values without inspecting env.sh.
    configuration = json.loads(json.dumps(config, default=str))
    configuration.update(slurm_job_id=job, slurm_array_task_id=task)
    _run = wandb.init(project=PROJECT, entity=os.environ.get('WANDB_ENTITY') or None,
                      mode=os.environ.get('WANDB_MODE', 'online'),
                      id=previous['id'] if previous else None, resume='allow' if previous else None,
                      name=f'{kind}-{job}' + (f'-{task}' if task else ''), job_type=kind,
                      group='parauto-' + hashlib.sha256(str(base).encode()).hexdigest()[:12],
                      dir=str(directory), config=configuration,
                      settings=wandb.Settings(console='redirect', init_timeout=120))
    state_path.write_text(json.dumps(dict(id=_run.id, url=_run.url, project=PROJECT))+'\n')
    _run.define_metric('train/step')
    _run.define_metric('train/*', step_metric='train/step')
    source = Path(__file__).resolve().parent
    _run.log_code(str(source), include_fn=lambda p: Path(p).suffix in ('.py', '.sh') and 'out' not in Path(p).parts)
    _started, _exit_code = time.perf_counter(), 0
    print(json.dumps({'wandb_run_url': _run.url, 'wandb_project': PROJECT}), flush=True)
    return _run


def start_cli(args, *, kind=None):
    """Explicit CLI entry-point call; argument values become W&B config."""
    options = vars(args)
    roots = []
    for key in ('output', 'run', 'root'):
        value = options.get(key)
        if value is not None:
            path = Path(value).resolve()
            roots.append(path.parent if path.suffix in SMALL_SUFFIXES else path)
    if options.get('phase') == 'cache' and options.get('cache'):
        roots.append(Path(options['cache']).resolve())
    kind = kind or '-'.join(filter(None, (Path(sys.argv[0]).stem, options.get('phase', ''))))
    base = roots[0] if roots else Path(os.environ.get('COFAC_DATA', '.')) / 'parauto' / 'wandb_runs' / os.environ.get('SLURM_JOB_ID', 'local') / kind
    return start(kind, base, options, resume=bool(options.get('resume')), extra_roots=roots[1:])


def configure(values=None, matrix_names=None):
    global _matrix_names
    if matrix_names is not None:
        _matrix_names = list(matrix_names)
    if _run is not None and values:
        _run.config.update(json.loads(json.dumps(values, default=str)), allow_val_change=True)


def log_metrics(metrics, *, prefix='', step=None):
    if _run is None:
        return
    values = flat_scalars(metrics, prefix)
    if step is not None:
        values['train/step'] = step
    if values:
        _run.log(values)


def log_train(row, matrix=None):
    index = row.get('matrix_index', 0)
    name = matrix or (_matrix_names[index] if index < len(_matrix_names) else str(index))
    values = {k: v for k, v in row.items() if k not in ('step', 'matrix_index', 'total_loss')}
    metrics = flat_scalars(values, f'train/{name}')
    if 'total_loss' in row:
        metrics['train/total_loss'] = row['total_loss']
    log_metrics(metrics, step=row['step'])


def log_initialization(matrix, info):
    log_metrics(info, prefix=f'initialization/{matrix}')


def log_result(value, path):
    """Called at the JSON write site, independently of printed messages."""
    if _run is not None:
        _run.summary.update(flat_scalars(value, f'results/{Path(path).name}'))


def finish(exit_code=None):
    global _run
    if _run is None:
        return
    run, _run = _run, None
    code = _exit_code if exit_code is None else exit_code
    run.summary.update({'invocation_seconds': time.perf_counter()-_started, 'process_exit_code': code})
    try:
        publish_outputs(run, _wandb, _roots, _before)
    except BaseException:
        code = 1
        raise
    finally:
        run.finish(exit_code=code)


def run_main(function):
    """Finish uploads explicitly, before Python/SDK shutdown tears down threads."""
    code = 0
    try:
        return function()
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        if code:
            traceback.print_exc()
        raise
    except BaseException:
        code = 1
        traceback.print_exc()
        raise
    finally:
        finish(code)
