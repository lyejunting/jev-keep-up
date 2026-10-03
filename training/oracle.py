from backend.app.schemas import GameState, Prediction


def target_x(state: GameState) -> float:
    # Top paddle center=48, height=14, radius=9: incoming contact at y=64.
    # Away-going balls are tracked until the opponent changes their trajectory.
    target = state.ball_x
    if state.ball_vy < -1e-9:
        seconds = max(0, (64 - state.ball_y) / state.ball_vy)
        target += state.ball_vx * seconds
    span = state.game_width - 18
    folded = (target - 9) % (2 * span)
    return 9 + (folded if folded <= span else 2 * span - folded)


class OraclePolicy:
    name = "oracle"

    def predict(self, state: GameState) -> Prediction:
        distance = target_x(state) - state.jev_x
        dead_zone = max(1, state.paddle_width * 0.15)
        return Prediction(prediction="STAY" if abs(distance) <= dead_zone else "LEFT" if distance < 0 else "RIGHT", confidence=1)
