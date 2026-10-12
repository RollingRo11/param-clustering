import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parauto_tracking as tracking
import pytest


def test_explicit_metrics_are_independent_of_console_text(monkeypatch, capsys):
    rows = []
    class Run:
        def log(self, row): rows.append(row)
    monkeypatch.setattr(tracking, '_run', Run())
    monkeypatch.setattr(tracking, '_matrix_names', ['layer0.mlp_in'])
    print('Any wording, including an unrelated JSON string: {"step": 999}')
    assert rows == []
    tracking.log_train(dict(step=10, matrix_index=0, loss=.2, total_loss=.9))
    assert rows == [{'train/step': 10, 'train/layer0.mlp_in/loss': .2, 'train/total_loss': .9}]
    tracking.log_train(dict(step=10, loss=.1), matrix='h.3.attn.q_proj')
    assert rows[-1]['train/h.3.attn.q_proj/loss'] == .1
    assert 'Any wording' in capsys.readouterr().out


def test_old_files_and_cache_tensors_are_not_uploaded(tmp_path):
    old = tmp_path / 'old.json'
    old.write_text('{}')
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache' / 'matrix_00.pt').write_bytes(b'cache')
    (tmp_path / '.wandb').mkdir()
    (tmp_path / '.wandb' / 'credentials.json').write_text('excluded')
    before = tracking.snapshot([tmp_path])
    assert set(before) == {str(old)}
    (tmp_path / 'pieces.pt').write_bytes(b'new model')
    (tmp_path / 'report.md').write_text('new report')
    after = tracking.snapshot([tmp_path])
    assert {p for p in after if p not in before} == {str(tmp_path / 'pieces.pt'), str(tmp_path / 'report.md')}
    assert {p.name for _, p in tracking.candidate_files([tmp_path])} == {'old.json', 'pieces.pt', 'report.md'}


def test_run_finishes_before_return_and_on_failure(monkeypatch):
    finished = []
    monkeypatch.setattr(tracking, 'finish', finished.append)
    assert tracking.run_main(lambda: 42) == 42
    assert finished == [0]
    def fail(): raise ValueError('expected validation failure')
    with pytest.raises(ValueError, match='expected validation failure'):
        tracking.run_main(fail)
    assert finished == [0, 1]
