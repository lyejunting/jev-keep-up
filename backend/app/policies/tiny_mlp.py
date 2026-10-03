import os
from pathlib import Path
from threading import Lock
import torch
from torch import nn
from .base import ACTIONS, FEATURES, NORMALIZATION, normalize
from ..schemas import GameState, Prediction
from ..inference import InferenceBusyError

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "keepup_mlp_v1.pt"


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, features):
        return self.layers(features)


class TinyMLPPolicy:
    name = "tiny_mlp"

    def __init__(self, path=None):
        self.path = Path(path or os.getenv("TINY_MLP_PATH", str(DEFAULT_MODEL)))
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        if (checkpoint.get("normalization") != NORMALIZATION or
                checkpoint.get("features") != list(FEATURES) or checkpoint.get("actions") != list(ACTIONS)):
            raise ValueError("Tiny MLP checkpoint has incompatible feature/action metadata")
        self.model = TinyMLP().eval()
        self.model.load_state_dict(checkpoint["state_dict"])
        self._lock = Lock()

    def predict(self, state: GameState) -> Prediction:
        if not self._lock.acquire(blocking=False):
            raise InferenceBusyError("Tiny MLP is still processing the previous prediction")
        try:
            with torch.inference_mode():
                probabilities = self.model(torch.tensor(normalize(state), dtype=torch.float32)).softmax(-1)
                if not torch.isfinite(probabilities).all():
                    raise ValueError("Tiny MLP returned nonfinite probabilities")
                best = int(probabilities.argmax())
                return Prediction(prediction=ACTIONS[best], confidence=float(probabilities[best]))
        finally:
            self._lock.release()
