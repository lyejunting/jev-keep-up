from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GameState(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    # Positions are paddle/ball centers in pixels; velocities are pixels/second.
    # The ball may briefly be outside the court when a point is scored.
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    jev_x: float
    paddle_width: float = Field(gt=0)
    game_width: float = Field(gt=0)
    game_height: float = Field(gt=0)


class Prediction(BaseModel):
    prediction: Literal["LEFT", "STAY", "RIGHT"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class PredictResponse(Prediction):
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    provider: str
