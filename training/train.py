import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from backend.app.schemas import GameState
from backend.app.policies.base import ACTIONS, FEATURES, NORMALIZATION, normalize
from backend.app.policies.tiny_mlp import TinyMLP


def load_data(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    states = [GameState(**{key: float(row[key]) for key in FEATURES}) for row in rows]
    x = torch.tensor([normalize(state) for state in states], dtype=torch.float32)
    y = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    if len(y) < 15 or set(y.tolist()) != {0, 1, 2}:
        raise ValueError("Dataset needs at least 15 samples and all three labels")
    return states, x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/training.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/keepup_mlp_v1.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    _, x, y = load_data(args.data)
    # Stratified, repeatable 80/20 split; evaluation uses an independent seed/dataset.
    train_indices, val_indices = [], []
    for label in range(3):
        indices = (y == label).nonzero().flatten()
        indices = indices[torch.randperm(len(indices))]
        count = max(1, int(len(indices) * 0.2))
        val_indices.extend(indices[:count].tolist())
        train_indices.extend(indices[count:].tolist())
    tx, ty = x[train_indices], y[train_indices]
    vx, vy = x[val_indices], y[val_indices]
    model = TinyMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = torch.nn.CrossEntropyLoss()
    best = (-1.0, float("inf"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(tx))
        for indices in order.split(args.batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(tx[indices]), ty[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_logits, val_logits = model(tx), model(vx)
            train_loss, val_loss = float(loss_fn(train_logits, ty)), float(loss_fn(val_logits, vy))
            train_accuracy = float((train_logits.argmax(1) == ty).float().mean())
            val_accuracy = float((val_logits.argmax(1) == vy).float().mean())
        print(f"epoch={epoch} training_loss={train_loss:.5f} validation_loss={val_loss:.5f} training_accuracy={train_accuracy:.4f} validation_accuracy={val_accuracy:.4f}")
        if val_accuracy > best[0] or (val_accuracy == best[0] and val_loss < best[1]):
            best = (val_accuracy, val_loss)
            metrics = {"epoch": epoch, "training_accuracy": train_accuracy, "validation_accuracy": val_accuracy,
                       "training_loss": train_loss, "validation_loss": val_loss, "samples": len(y),
                       "train_samples": len(ty), "validation_samples": len(vy), "seed": args.seed,
                       "architecture": "8-32-ReLU-32-ReLU-3", "parameters": sum(p.numel() for p in model.parameters())}
            torch.save({"state_dict": model.state_dict(), "features": list(FEATURES), "actions": list(ACTIONS),
                        "normalization": NORMALIZATION, "metrics": metrics}, args.output)
            args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("Best checkpoint:", args.output)
    print(args.output.with_suffix(".json").read_text())


if __name__ == "__main__":
    main()
