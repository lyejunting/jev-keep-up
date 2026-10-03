# JEV Keep Up

React + TypeScript + Canvas, with a FastAPI decision service. Human paddle at the bottom, JEV at the top. First to 10 wins.

Start the whole program from the project root:

```sh
make start
```

The command automatically stops any process listening on ports 8000 or 5173, waits up to 10 seconds for the ports to become free, then starts both services. To free the ports without restarting, run `make stop`.

Open http://127.0.0.1:5173/ once the services are ready. Press Ctrl+C to stop both services. If either service exits, the script stops the other. The first start downloads the model into the ignored `.cache/huggingface/`.

The launcher checks Node compatibility before starting either service and prints the selected version. If your terminal selects an incompatible Node, it also checks the standard Homebrew locations. To choose an executable explicitly, run `NODE_BIN=/opt/homebrew/bin/node make start` (adjust the path for your installation). Errors such as `crypto.getRandomValues is not a function` indicate an incompatible runtime.

One-time setup (Make, Python 3.10+ and Node.js 20.19+ or 22.12+):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
npm --prefix frontend install
```

Use Left/Right or A/D. Start begins a match; Restart starts a fresh match. Ball speed changes immediately. Touch devices also get hold-to-move controls.

```sh
npm run typecheck
npm test
npm run build
```

The frontend sends structured game state to `POST http://127.0.0.1:8000/predict`, with a 100 ms minimum interval between request starts and only one request pending. Actual inference frequency depends on model latency. The last valid decision remains active while a prediction runs; requests time out after 30 seconds. Backend failures clear JEV to `STAY` and show Unavailable; the match continues and inference retries automatically. No images or base64 are sent. Predictions from a previous round or restart are discarded. The backend rejects overlapping forward passes rather than queuing them, including work still running after an HTTP cancellation.

To configure the endpoint or inference interval, copy `frontend/.env.example` to `frontend/.env.local`, edit the values, and restart Vite. The backend allows local Vite origins on port 5173.

`JEV_PROVIDER=openjev` is the default. `OpenJevProvider` loads [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev), subfolder `qwen3.5-0.8b-nli-v2s-long`, once at backend startup. The revision is pinned to `a20448012c213128955ca0c693e7c943865cab77`. It uses the repository's `config.nli_template`, right padding, attention-mask last-token pooling, and three NLI claims for LEFT/STAY/RIGHT. Confidence is the chosen entailment probability normalized over those three candidates, as in the repository's decision adapter; it is not a calibrated estimate of paddle accuracy. There is no text generation or hosted inference.

PyTorch chooses CUDA, native Mac Metal (`mps`), or CPU. Float32 is used for this local integration. Set `OPENJEV_DEVICE=cpu` to select CPU explicitly, or `OPENJEV_CACHE_DIR` to change the cache directory. Startup logs report the device and measured model load time. Model/load failures never silently switch to mock.

OpenJEV now uses the repository's `predict_hypotheses` shared-prefix approach: one prefix prefill, then three isolated cached continuations in a batch. The previous implementation already used one three-action batch, rather than three separate forwards. Prompts, checkpoint, float32 precision, and confidence calculation are unchanged. Inference uses evaluation mode without gradients and transfers the final three probabilities to CPU once.

Successful `/predict` responses include a `Server-Timing` header for `request_parsing`, `premise`, `tokenization`, `model_forward`, `post_processing`, and `total` (milliseconds). Parsing measures body reading and JSON decoding; validation, worker dispatch, and response serialization are included in total. Tokenization includes input preparation/device upload. GPU synchronization separates forward time from post-processing; total measures the FastAPI route, excluding network transit.

Local MPS benchmark: 24 identical, varied game states per implementation after five warmups, through FastAPI's in-process HTTP client. p95 uses the nearest-rank percentile.

| Timing | Previous batch | Shared prefix |
| --- | ---: | ---: |
| Total median | 277.69 ms | 212.37 ms |
| Total p95 | 326.47 ms | 228.41 ms |
| Tokenization median | 1.43 ms | 2.03 ms |
| Model forward median | 275.37 ms | 209.34 ms |

All 24 decisions were unchanged; maximum confidence difference was 0.00000301 from floating-point execution order. These measurements describe this machine and workload, not a guaranteed inference rate.

For development without model inference:

```sh
JEV_PROVIDER=mock make start
```

The UI identifies the actual provider as **OpenJEV-style 0.8B** or **Mock** and retains its identity if inference becomes unavailable. `MockJevProvider` uses a deterministic paddle-relative dead zone and fixed heuristic confidence.

The endpoint accepts JSON with `ball_x`, `ball_y`, `ball_vx`, `ball_vy`, `jev_x`, `paddle_width`, `game_width`, and `game_height`. Positions are centers in court pixels; velocities are pixels/second, including the selected ball-speed multiplier (zero while a rally is paused). Responses contain `prediction`, `confidence`, server-side `latency_ms`, and `provider`. The browser owns physics, scoring, rendering, and human controls; Python only selects JEV decisions.

Backend checks (with the virtual environment active, from `backend/`):

```sh
python -m compileall -q app
python -m unittest discover -s tests
```
