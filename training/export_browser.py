"""Export the trained checkpoint as a small, committed browser model asset."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from backend.app.policies.base import ACTIONS, FEATURES, NORMALIZATION
from backend.app.policies.tiny_mlp import TinyMLPPolicy, DEFAULT_MODEL

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'frontend/src/models/keepup_mlp_v1.json'


def export(path, output):
    policy = TinyMLPPolicy(path)
    layers = []
    for layer in policy.model.layers:
        if isinstance(layer, torch.nn.Linear):
            if not torch.isfinite(layer.weight).all() or not torch.isfinite(layer.bias).all():
                raise ValueError('Checkpoint contains nonfinite weights')
            layers.append({'weights': layer.weight.detach().tolist(), 'bias': layer.bias.detach().tolist()})
    asset = {'version': 1, 'normalization': NORMALIZATION, 'features': list(FEATURES),
             'actions': list(ACTIONS), 'architecture': [8, 32, 32, 3], 'parameters': 1443,
             'checkpoint_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'layers': layers}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asset, separators=(',', ':'), allow_nan=False) + '\n')
    print(f'Exported {output} ({output.stat().st_size} bytes, 1443 parameters)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.model, args.output)


if __name__ == '__main__': main()
