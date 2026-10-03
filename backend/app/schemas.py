from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class NormalizedRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    model: Literal["jev", "tiny_mlp"]
    features: list[float] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def positive_dimensions(self):
        if self.features[5] <= 0 or self.features[6] <= 0 or self.features[7] <= 0:
            raise ValueError("Paddle and court dimensions must be positive")
        return self
