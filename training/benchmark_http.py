"""Compare both deployed policies on identical states using the UI endpoints."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.schemas import GameState
from backend.app.policies.base import FEATURES, normalize
from training.generate_data import generate
from training.evaluate import percentiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, default=Path('models/http-benchmark.json'))
    args = parser.parse_args()
    rows, _ = generate(args.samples, args.seed)
    states = [GameState(**dict(zip(FEATURES, row[:8]))) for row in rows]
    reports = []
    for model in ('jev', 'tiny_mlp'):
        inference, roundtrip, decisions = [], [], []
        for i, state in enumerate(states[:5] + states):
            payload = state.model_dump() if model == 'jev' else {'model': model, 'features': normalize(state)}
            endpoint = '/predict' if model == 'jev' else '/predict/normalized'
            request = Request(args.url + endpoint, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            started = perf_counter()
            with urlopen(request, timeout=60) as response:
                data = json.load(response)
            elapsed = (perf_counter() - started) * 1000
            if i >= 5:
                inference.append(data['latency_ms'])
                roundtrip.append(elapsed)
                decisions.append(data['prediction'])
        reports.append({'model': model, 'provider': data['provider'], 'samples': len(states), 'seed': args.seed,
                        'warmups': 5, 'server_inference_latency': percentiles(inference),
                        'http_roundtrip_latency': percentiles(roundtrip), 'decisions': decisions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + '\n')
    print(args.output.read_text())


if __name__ == '__main__': main()
