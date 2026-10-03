from time import perf_counter
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import InferenceBusyError, JevProvider, create_provider
from .schemas import GameState, PredictResponse, NormalizedRequest
from .policies import POLICY_FACTORIES, POLICY_LABELS
from threading import Lock
from typing import Literal
from .profiling import PredictTimingRoute

provider: JevProvider | None = None
policies = {}
policy_errors = {}
policy_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider
    # Load once before accepting requests. Loading failures remain explicit;
    # there is no automatic substitution of mock or a different checkpoint.
    provider = create_provider()
    policies.clear()
    policy_errors.clear()
    # Tiny MLP is independent: a missing checkpoint cannot prevent Jev startup.
    try:
        policies["tiny_mlp"] = POLICY_FACTORIES["tiny_mlp"]()
    except Exception as exc:
        policy_errors["tiny_mlp"] = str(exc)
    yield
    provider = None
    policies.clear()


app = FastAPI(title="JEV Keep Up inference", lifespan=lifespan)
app.router.route_class = PredictTimingRoute
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


@app.post("/predict", response_model=PredictResponse)
def predict(state: GameState, model: Literal["jev", "tiny_mlp"] = "jev") -> PredictResponse:
    # FastAPI runs this synchronous provider call in its worker thread pool.
    started = perf_counter()
    selected = provider if model == "jev" else policies.get(model)
    if selected is None and model == "tiny_mlp":
        # Allow training a checkpoint while the backend is already running.
        with policy_lock:
            selected = policies.get(model)
            if selected is None:
                try:
                    selected = policies[model] = POLICY_FACTORIES[model]()
                    policy_errors.pop(model, None)
                except Exception as exc:
                    policy_errors[model] = str(exc)
    if selected is None:
        raise HTTPException(status_code=503, detail=policy_errors.get(model, "JEV provider has not loaded"))
    try:
        prediction = selected.predict(state)
    except InferenceBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logging.getLogger("uvicorn.error").exception("%s inference failed", model)
        raise HTTPException(status_code=503, detail=f"{model} inference failed") from exc
    return PredictResponse(
        **prediction.model_dump(),
        latency_ms=round((perf_counter() - started) * 1000, 2),
        provider=selected.name,
    )


@app.get("/policies")
def list_policies():
    return [{"id": model, "label": POLICY_LABELS[model],
             "available": provider is not None if model == "jev" else model in policies,
             "error": policy_errors.get(model)} for model in POLICY_FACTORIES]


@app.post("/predict/normalized", response_model=PredictResponse)
def predict_normalized(request: "NormalizedRequest"):
    # Browser supplies the exact training representation; reconstruct the raw
    # state so every policy shares predict(GameState), including existing Jev.
    f = request.features
    width, height = f[6] * 960, f[7] * 640
    state = GameState(ball_x=f[0] * width, ball_y=f[1] * height,
                      ball_vx=f[2] * width, ball_vy=f[3] * height,
                      jev_x=f[4] * width, paddle_width=f[5] * width,
                      game_width=width, game_height=height)
    return predict(state, request.model)
