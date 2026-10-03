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

Use Left/Right or A/D. Start begins a match; Restart starts a fresh match. Pause/Resume or Space freezes and resumes the same rally, including the point timer. Space keeps native behavior in form controls and buttons. Ball speed changes immediately. Touch devices also get hold-to-move controls.

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

The model selector offers **Jev 0.8B** and **Tiny MLP**; development Jev is explicitly labeled **Mock** and retains its identity if inference becomes unavailable. `MockJevProvider` uses a deterministic paddle-relative dead zone and fixed heuristic confidence.

The endpoint accepts JSON with `ball_x`, `ball_y`, `ball_vx`, `ball_vy`, `jev_x`, `paddle_width`, `game_width`, and `game_height`. Positions are centers in court pixels; velocities are pixels/second, including the selected ball-speed multiplier. Responses contain `prediction`, `confidence`, server-side `latency_ms`, and `provider`. The browser owns physics, scoring, rendering, and human controls; Python selects the chosen policy’s decisions. Inference requests run only during visible, active rallies. Pausing aborts the pending browser request and discards its response; a server forward already in progress may finish, but cannot alter the paused game. Resume keeps one existing animation loop.

Backend checks (with the virtual environment active, from `backend/`):

```sh
python -m compileall -q app
PYTHONPATH=..:. python -m unittest discover -s tests
```


## Tiny MLP training and evaluation

Run from the repository root with the existing virtual environment (PyTorch is
already in `backend/requirements.txt`):

```sh
.venv/bin/python training/generate_data.py --samples 100000 --seed 42
.venv/bin/python training/train.py --epochs 80
.venv/bin/python training/evaluate.py --samples 3000 --games 100 --simulate-latency
.venv/bin/python training/evaluate.py --games 100 --random-start --simulate-latency --output models/stress-evaluation.json
.venv/bin/python training/benchmark_http.py --samples 100
```

The HTTP benchmark requires a running backend and compares identical seeded
states after five warmups per policy, using the UI's actual endpoints. It reports
server inference and HTTP round-trip median and nearest-rank p95 separately.
Model load time and the frontend's 100 ms scheduling interval are excluded.

`training/simulator.py` mirrors the current 960×640 court, radius 9, paddle
width 128, speeds 240/560, ball speed 330, 240 Hz physics steps, wall/paddle
collisions, 0.9-second point timer and first-to-10 scoring. Seeded RNG replaces
`Math.random`. The top oracle predicts contact at y=64, folds wall bounces,
and uses a 15%-paddle-width dead zone. Away-going balls are tracked until the
opponent changes their trajectory. Geometry follows this game's fixed paddles.
The faster bottom paddle is controlled by the symmetric oracle in evaluations.

The generator mixes randomized scenarios and short physics rollouts and uses
rejection sampling to balance LEFT/STAY/RIGHT without changing oracle labels.
CSV files contain raw state and integer labels (0/1/2); training normalizes them.
The eight features, in order, are ball x/width, ball y/height, vx/width, vy/height,
paddle center/width, paddle width/court width, court width/960, court height/640.
Positions are centers and velocities are pixels/second. The browser implements
the same normalization for Tiny MLP and submits it to `/predict/normalized`.
Jev's existing raw-state `/predict` contract is preserved. Both implementations
conform to `Policy.predict(GameState) -> Prediction`; register new factories in
`backend/app/policies/` and add their UI options without changing physics.

The network is 8→32→32→3 with ReLU hidden activations (1,443 parameters).
Training uses Adam, CrossEntropyLoss, a seeded stratified 80/20 split and CPU
execution. The checkpoint with best validation accuracy (validation loss breaks
ties) is saved with feature order, action labels and normalization version.
Evaluation generates an independent dataset by default; pass `--data` only for
an independent test CSV. Per-class accuracy and the confusion matrix accompany
match scores, return counts, miss rate and timeout counts. A zero score against
the oracle can indicate an ongoing rally; inspect returns and completed matches.
`--simulate-latency` delays actions by measured median inference time; it models
a fixed delay rather than jitter or HTTP overhead. `--compare-jev` additionally
runs real Jev on the same states and game seeds (default 30 states, 3 games;
adjust `--jev-samples`, `--jev-games`, `--seconds` for expensive runs).

Datasets, checkpoints, logs and evaluation artifacts in `data/` and `models/`
are ignored by Git. The locally trained checkpoint is
`models/keepup_mlp_v1.pt`; override its location with `TINY_MLP_PATH`.
Missing/incompatible checkpoints return unavailable explicitly, with no policy
substitution. A checkpoint created after backend startup loads on the first
Tiny MLP request; restart the backend to load replacements of an existing model.
`GET /policies` reports backend availability.

Verification:

```sh
PYTHONPATH=backend:. .venv/bin/python -m unittest discover -s backend/tests
npm --prefix frontend test
npm --prefix frontend run build
```

An optional real-browser smoke check is `frontend/tests/browser-smoke.cjs`.
With both services running, install Playwright in a temporary directory and run:

```sh
npm install --prefix /tmp/keepup-browser playwright
PLAYWRIGHT_MODULE=/tmp/keepup-browser/node_modules/playwright CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node frontend/tests/browser-smoke.cjs
```

It checks both real policies, frozen canvas and requests during pause, switching
while paused, Space and a single game animation loop. Its screenshot goes to
ignored `models/browser-smoke.png`.
