"""End-to-end flow on dummy_dataset: train.py -> evaluate.py -> infer.py (few epochs)."""

import csv
import json
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, 'dummy_dataset')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _run(*args):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    result = subprocess.run([sys.executable, *args, '--device', DEVICE], cwd=ROOT, env=env,
                            capture_output=True, text=True, encoding='utf-8', errors='replace')
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result.stdout


def test_train_evaluate_infer(tmp_path):
    runs = tmp_path / 'runs'
    _run('train.py', '--dataset', DATASET, '--classes', 'dummy_class', '--image_epochs', '1', '1',
         '--pixel_epochs', '1', '--output_dir', str(runs))
    (run_dir,) = list(runs.iterdir())
    class_dir = run_dir / 'dummy_class'
    for name in ('image_head.pt', 'pixel_head.pt', 'metrics.json'):
        assert (class_dir / name).is_file(), name

    # Dataset without a planned recipe: fallback with the ablation stored in the checkpoint.
    ckpt = torch.load(class_dir / 'pixel_head.pt', map_location='cpu', weights_only=False)
    assert ckpt['training_recipe']['name'] == 'fallback'
    assert ckpt['level_ablation']['status'] == 'done'
    assert ckpt['inference_recipe']['levels'] == ckpt['level_ablation']['selected']
    trained = json.loads((class_dir / 'metrics.json').read_text(encoding='utf-8'))

    _run('evaluate.py', '--dataset', DATASET, '--classes', 'dummy_class', '--model_dir',
         str(run_dir), '--output', str(tmp_path / 'eval'))
    (eval_dir,) = list((tmp_path / 'eval').iterdir())
    with open(eval_dir / 'summary.csv', encoding='utf-8') as fh:
        row = next(csv.DictReader(fh))
    assert abs(float(row['pixel_auroc']) - trained['pixel_auroc']) < 1e-6

    _run('infer.py', '--classes', 'dummy_class', '--model_dir', str(run_dir), '--input',
         os.path.join(DATASET, 'dummy_class', 'test', 'anomaly'), '--output', str(tmp_path / 'inf'))
    out = tmp_path / 'inf' / 'dummy_class'
    assert len(list((out / 'heatmaps').glob('*.npy'))) == 8
    assert (out / 'scores.csv').is_file()
