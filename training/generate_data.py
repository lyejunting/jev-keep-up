import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.policies.base import ACTIONS, FEATURES
from training.oracle import OraclePolicy
from training.simulator import Simulator


def generate(samples, seed):
    if samples < 3:
        raise ValueError("samples must be at least 3")
    sim, oracle = Simulator(seed), OraclePolicy()
    rng = random.Random(seed)
    quotas = [samples // 3 + int(i < samples % 3) for i in range(3)]
    counts = [0, 0, 0]
    rows = []
    attempts = 0
    # Rejection sampling retains actual oracle labels, with balanced classes.
    while len(rows) < samples:
        attempts += 1
        if attempts > samples * 1000:
            raise RuntimeError("Unable to fill class quotas")
        sim.randomize()
        # Half of candidates follow short physics rollouts (including bounces).
        for _ in range(rng.randrange(0, 120) if rng.random() < 0.5 else 0):
            sim.step(oracle.predict(sim.state()).prediction)
            if sim.timer > 0:
                break
        state = sim.state()
        label = ACTIONS.index(oracle.predict(state).prediction)
        if counts[label] < quotas[label]:
            rows.append([getattr(state, key) for key in FEATURES] + [label])
            counts[label] += 1
    rng.shuffle(rows)
    return rows, counts


def main():
    parser = argparse.ArgumentParser(description="Generate balanced raw game states with oracle labels")
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/training.csv"))
    args = parser.parse_args()
    rows, counts = generate(args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*FEATURES, "label"])
        writer.writerows(rows)
    metadata = {"samples": len(rows), "seed": args.seed, "distribution": dict(zip(ACTIONS, counts))}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
