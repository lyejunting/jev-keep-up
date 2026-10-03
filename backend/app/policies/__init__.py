"""Add a policy here without changing frontend physics or the prediction route."""
from .base import Policy
from .jev import create_provider
from .tiny_mlp import TinyMLPPolicy

POLICY_FACTORIES = {"jev": create_provider, "tiny_mlp": TinyMLPPolicy}
POLICY_LABELS = {"jev": "Jev 0.8B", "tiny_mlp": "Tiny MLP"}
