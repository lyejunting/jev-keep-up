import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from backend.app.policies.base import ACTIONS
from backend.app.policies.tiny_mlp import TinyMLPPolicy, DEFAULT_MODEL
from backend.app.inference import create_provider
from training.generate_data import generate
from training.train import load_data
from training.oracle import OraclePolicy
from training.simulator import play
from backend.app.policies.base import FEATURES
from backend.app.schemas import GameState


def percentiles(times):
    ordered = sorted(times)
    return {"median_ms": statistics.median(times), "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1]}


def evaluate(policy, states, labels, games, seed, seconds, interval, delay, random_start=False):
    for state in states[:5]:
        policy.predict(state)
    confusion = [[0] * 3 for _ in range(3)]
    times = []
    for state, label in zip(states, labels):
        started = perf_counter()
        result = policy.predict(state)
        times.append((perf_counter() - started) * 1000)
        confusion[label][ACTIONS.index(result.prediction)] += 1
    latency = percentiles(times)
    effective_delay = latency["median_ms"] if delay else 0
    results = [play(policy, seed + i, seconds, interval, effective_delay, random_start=random_start) for i in range(games)]
    scores = [result["score"] for result in results]
    misses, hits = sum(r["misses"] for r in results), sum(r["hits"] for r in results)
    return {"model": policy.name, "classification_samples": len(states),
            "classification_accuracy": sum(confusion[i][i] for i in range(3)) / len(states),
            "class_accuracy": {action: confusion[i][i] / sum(confusion[i]) if sum(confusion[i]) else None for i, action in enumerate(ACTIONS)},
            "confusion_matrix_rows_true_columns_predicted": confusion, "inference_latency": latency,
            "games": games, "average_returns": statistics.mean(r["hits"] for r in results),
            "average_opponent_score": statistics.mean(r["opponent_score"] for r in results),
            "randomized_opening": random_start, "average_score": statistics.mean(scores), "median_score": statistics.median(scores),
            "best_score": max(scores), "miss_rate_percent": 100 * misses / max(1, misses + hits),
            "completed_games": sum(r["completed"] for r in results), "max_game_seconds": seconds,
            "opponent": "oracle, human paddle speed 560", "decision_interval_ms": interval,
            "simulated_inference_delay_ms": effective_delay, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description="Independent classification and actual simulated matches")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, help="Independent test CSV; do not reuse training CSV")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--interval-ms", type=float, default=100)
    parser.add_argument("--random-start", action="store_true", help="Stress test with randomized opening positions and velocities")
    parser.add_argument("--simulate-latency", action="store_true")
    parser.add_argument("--compare-jev", action="store_true", help="Run real Jev on the same test states and game seeds (can be slow)")
    parser.add_argument("--jev-samples", type=int, default=30)
    parser.add_argument("--jev-games", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("models/evaluation.json"))
    args = parser.parse_args()
    if min(args.samples, args.games, args.seconds, args.interval_ms, args.jev_samples, args.jev_games) <= 0:
        parser.error("sample counts, game counts, seconds and interval must be positive")
    torch.set_num_threads(1)
    if args.data:
        states, _, labels = load_data(args.data)
        labels = labels.tolist()
    else:
        rows, _ = generate(args.samples, args.seed)
        states = [GameState(**dict(zip(FEATURES, row[:8]))) for row in rows]
        labels = [row[8] for row in rows]
    mlp = TinyMLPPolicy(args.model)
    reports = [evaluate(mlp, states, labels, args.games, args.seed, args.seconds, args.interval_ms, args.simulate_latency, args.random_start),
               evaluate(OraclePolicy(), states, labels, args.games, args.seed, args.seconds, args.interval_ms, False, args.random_start)]
    if args.compare_jev:
        jev = create_provider("openjev")
        # Same states for latency/decision comparison; separate limited game count.
        subset, subset_labels = states[:args.jev_samples], labels[:args.jev_samples]
        reports.append(evaluate(mlp, subset, subset_labels, args.jev_games, args.seed, args.seconds, args.interval_ms, args.simulate_latency, args.random_start))
        reports.append(evaluate(jev, subset, subset_labels, args.jev_games, args.seed, args.seconds, args.interval_ms, args.simulate_latency, args.random_start))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
