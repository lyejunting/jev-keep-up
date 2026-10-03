from time import perf_counter
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import InferenceBusyError, JevProvider, create_provider
from .schemas import GameState, PredictResponse
from .profiling import PredictTimingRoute

provider: JevProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider
    # Load once before accepting requests. Loading failures remain explicit;
    # there is no automatic substitution of mock or a different checkpoint.
    provider = create_provider()
    yield
    provider = None


app = FastAPI(title="JEV Keep Up inference", lifespan=lifespan)
app.router.route_class = PredictTimingRoute
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.post("/predict", response_model=PredictResponse)
def predict(state: GameState) -> PredictResponse:
    # FastAPI runs this synchronous provider call in its worker thread pool.
    started = perf_counter()
    if provider is None:
        raise HTTPException(status_code=503, detail="JEV provider has not loaded")
    try:
        prediction = provider.predict(state)
    except InferenceBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logging.getLogger("uvicorn.error").exception("JEV inference failed")
        raise HTTPException(status_code=503, detail="JEV inference failed") from exc
    return PredictResponse(
        **prediction.model_dump(),
        latency_ms=round((perf_counter() - started) * 1000, 2),
        provider=provider.name,
    )
