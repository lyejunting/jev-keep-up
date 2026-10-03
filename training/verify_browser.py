"""Compare exported TypeScript inference against PyTorch on independent states.

First run npm --prefix frontend run test:compile to compile the TypeScript modules.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from backend.app.schemas import GameState
from backend.app.policies.base import ACTIONS, FEATURES, normalize
from backend.app.policies.tiny_mlp import TinyMLPPolicy, DEFAULT_MODEL
from training.generate_data import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--samples', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--node', default='node')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/browser-parity.json')
    parser.add_argument('--fixtures', type=Path, help='Save 36 reference cases for frontend regression tests')
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = TinyMLPPolicy(args.model)
    rows, _ = generate(args.samples, args.seed)
    states = [GameState(**dict(zip(FEATURES, row[:8]))) for row in rows]
    with torch.inference_mode():
        expected_logits = policy.model(torch.tensor([normalize(s) for s in states], dtype=torch.float32))
        expected_probabilities = expected_logits.softmax(-1)
    module = str(ROOT / 'frontend/tests/.compiled/services/tinyMlp.js')
    script = '''const fs = require('node:fs');
const { browserLogits, predictTinyMlp } = require(process.argv[1]);
const states = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(states.map(state => ({logits: browserLogits(state), ...predictTinyMlp(state)}))));'''
    process = subprocess.run([args.node, '-e', script, module], input=json.dumps([s.model_dump() for s in states]),
                             text=True, capture_output=True, check=True)
    actual = json.loads(process.stdout)
    expected_actions = expected_logits.argmax(-1).tolist()
    mismatches = sum(result['prediction'] != ACTIONS[label] for result, label in zip(actual, expected_actions))
    max_logits_error = max(abs(value - actual[i]['logits'][j]) for i, row in enumerate(expected_logits.tolist()) for j, value in enumerate(row))
    max_confidence_error = max(abs(actual[i]['confidence'] - float(expected_probabilities[i, label])) for i, label in enumerate(expected_actions))
    report = {'samples': len(states), 'seed': args.seed, 'action_agreement_percent': 100*(1-mismatches/len(states)),
              'action_mismatches': mismatches, 'max_logits_absolute_error': max_logits_error,
              'max_confidence_absolute_error': max_confidence_error,
              'browser_oracle_accuracy': sum(result['prediction'] == ACTIONS[row[8]] for result, row in zip(actual, rows))/len(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if args.fixtures:
        cases = [{'state': state.model_dump(), 'logits': logits, 'prediction': ACTIONS[label],
                  'confidence': float(expected_probabilities[i, label])}
                 for i, (state, logits, label) in enumerate(zip(states[:36], expected_logits.tolist()[:36], expected_actions[:36]))]
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        args.fixtures.write_text(json.dumps(cases, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if mismatches or max_logits_error > 2e-4 or max_confidence_error > 2e-5:
        raise SystemExit('Browser/PyTorch parity check failed')


if __name__ == '__main__': main()
