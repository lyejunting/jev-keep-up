from typing import Protocol
from ..schemas import GameState, Prediction

ACTIONS = ("LEFT", "STAY", "RIGHT")
FEATURES = ("ball_x", "ball_y", "ball_vx", "ball_vy", "jev_x", "paddle_width", "game_width", "game_height")
NORMALIZATION = "court-relative-v1"


class Policy(Protocol):
    name: str

    def predict(self, state: GameState) -> Prediction: ...


def normalize(state: GameState) -> list[float]:
    # Velocities are court lengths/second. Dimensions retain scale information.
    return [state.ball_x / state.game_width, state.ball_y / state.game_height,
            state.ball_vx / state.game_width, state.ball_vy / state.game_height,
            state.jev_x / state.game_width, state.paddle_width / state.game_width,
            state.game_width / 960, state.game_height / 640]
