# Keep-Up implementation and measured results

Measured locally on 2026-10-03. Jev uses Apple Metal (MPS), float32, the original
pinned checkpoint and shared-prefix adapter. Tiny MLP runs on CPU.

## Latency benchmark

100 identical seeded states per policy, five warmups each, sequential local HTTP
requests and no concurrent game inference. Seed 2026. p95 is nearest rank.

| Model | Server median | Server p95 | HTTP round-trip median | HTTP round-trip p95 |
| --- | ---: | ---: | ---: | ---: |
| Jev 0.8B | 207.00 ms | 214.63 ms | 208.53 ms | 216.15 ms |
| Tiny MLP | 0.04 ms | 0.06 ms | 0.66 ms | 0.98 ms |

Server latency measures `predict` including state preprocessing, tensor creation,
forward pass and confidence extraction. It excludes model loading, HTTP
transport and the browser's 100 ms scheduling interval; the API rounds it to
0.01 ms. Jev's CPU probability copy synchronizes Metal before returning.
HTTP round-trip includes serialization, request handling and response reading.
These are measurements on this machine, not guaranteed rates elsewhere.

Reproduce with both services running:

```sh
.venv/bin/python training/benchmark_http.py --samples 100
```

Detailed aggregate measurements are retained in `training/results.json`.
The full local benchmark is `models/http-benchmark.json`.

## Jev 0.8B versus Tiny MLP success rates

Two success measures are available from the existing comparison: choosing the
oracle's action and returning an incoming ball. Match win rate requires completed
matches and is not yet available.

| Success measure | Jev 0.8B | Our Tiny MLP | Evaluation size |
| --- | ---: | ---: | --- |
| Oracle action agreement | 46.7% (14/30) | 80.0% (24/30) | Same 30 independent states |
| Paddle return success | 66.7% (4/6) | 100.0% (6/6) | One 20-second simulation per policy, same opening seed |
| Completed-match win rate | Not available | Not available | Neither match reached 10 points |

Oracle action agreement is correct predictions / evaluated states. Paddle return
success is hits / (hits + misses), equivalent to 100% minus miss rate. Each policy
plays separately against the same bottom oracle; this is not a head-to-head match
between Jev and Tiny MLP. Both use a 100 ms decision interval with their measured
median inference delay. Their subsequent trajectories differ as their actions
change paddle collisions.

These results favor Tiny MLP in this small comparison, but six paddle encounters
per policy are too few to establish a reliable gameplay success rate. Tiny MLP's
larger independent test scored 89.37% oracle agreement over 3,000 states and
99.33% paddle return success across 100 randomized-opening simulations. Jev has
not been evaluated at those larger sample sizes, so those figures should not be
compared directly with its small-sample results. A stronger comparison should run
both policies over the same larger set of seeds and report completed wins,
losses and timeouts separately.

## Architecture and behavior

- Existing ball motion, collision rules, paddle speeds and scoring are preserved.
- Engine pause saves PLAYING/POINT_SCORED and freezes all updates and the point
  timer. Resume continues that state. Restart clears the paused state.
- Pause and Space suspend inference requests, abort the browser request and
  invalidate late results. A backend forward already running may finish;
  cancellation cannot suspend a PyTorch forward. It cannot change the paused
  game or queue a second forward for the same policy.
- One canvas animation loop remains active for rendering while paused; repeated
  pause/resume never starts another loop. Hidden documents issue no inference.
- Policy selection resets only cached AI decisions, preserving match state.
  Backend `Policy.predict(GameState) -> Prediction` and frontend
  `PolicyController.decide()` separate inference from game physics.
- Jev remains in `backend/app/inference.py`, re-exported from `policies/jev.py`.
  Its prompt, model, confidence adapter and caching are unchanged.
- Tiny MLP uses a versioned eight-feature normalization shared by Python training
  and TypeScript inference. `/predict/normalized` carries those features.
  The original raw `/predict` endpoint continues to default to Jev.
- Checkpoints validate feature/action metadata. Missing weights are reported as
  unavailable without substitution. The backend can load newly created Tiny
  weights on its first request. Replacing already-loaded weights needs a restart.
- The deterministic oracle folds predicted top-paddle contact x through wall
  bounces. It is independent of the network. The seeded simulator mirrors the
  frontend rules and controls the bottom paddle with a symmetric oracle.

## Training

30,000 samples, seed 42, exactly 10,000 LEFT / 10,000 STAY / 10,000 RIGHT.
Candidates combine randomized numerical states and short physics rollouts;
rejection sampling balances classes without modifying labels.

Network: 8 → 32 + ReLU → 32 + ReLU → 3, **1,443 parameters**.
Adam (learning rate 0.002), CrossEntropyLoss, batch size 256, 100 CPU epochs.
Seeded stratified split: 24,000 training / 6,000 validation examples.
Best checkpoint: epoch 93, selected by validation accuracy with loss as tie-breaker.

| Metric | Result |
| --- | ---: |
| Training accuracy | 88.92% |
| Validation accuracy | 88.07% |
| Training loss | 0.27611 |
| Validation loss | 0.29089 |
| Independent test accuracy (3,000 samples, seed 2026) | 89.37% |
| LEFT / STAY / RIGHT test accuracy | 88.10% / 89.70% / 90.30% |

Confusion matrix (rows true LEFT/STAY/RIGHT; columns predicted LEFT/STAY/RIGHT):

```text
881   89   30
 28  897   75
  9   88  903
```

Weights: `models/keepup_mlp_v1.pt`; metadata: `models/keepup_mlp_v1.json`.
Datasets, weights and local logs are ignored by Git; the files exist locally.

## Gameplay evaluation

100 seeds, at most 120 simulated seconds per game, first-to-10, original paddle
speeds, 100 ms decision interval, faster bottom oracle opponent. Tiny actions
are delayed by measured median direct inference latency (~0.014 ms). Delays
are fixed medians and do not model HTTP overhead or jitter.

| Opening / policy | Average score | Median score | Best score | Average returns | Miss rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard serves / Tiny MLP | 0 | 0 | 0 | 36.03 | 0.00% |
| Standard serves / Oracle | 0 | 0 | 0 | 36.79 | 0.00% |
| Randomized opening / Tiny MLP | 0.12 | 0 | 1 | 35.74 | 0.67% |
| Randomized opening / Oracle | 0.12 | 0 | 1 | 36.36 | 0.60% |

No match reached 10 points within 120 seconds. Zero scores in standard openings
reflect sustained rallies, not completed wins. Miss rate is top-paddle misses /
(top-paddle hits + misses). Randomized openings can include initially difficult
or unreachable trajectories; they are a stress test rather than normal serves.

A limited real-Jev comparison used 30 identical test states and one identical
20-second randomized game seed per policy. Tiny had 6 returns, zero misses and
score 0; Jev had 4 returns, 2 misses and score 0 (33.3% miss rate). Neither match
completed. On that small classification subset, Tiny matched the oracle 80.0%
and Jev 46.7%. This single short game is insufficient for broad gameplay claims.
The direct comparison measured Jev 208.81 ms median / 269.73 ms p95; the larger
100-request HTTP benchmark above is the primary latency comparison.

## Verification

- All 19 frontend tests pass, including pause physics, point timers, restart,
  inference cancellation, response invalidation and model switching.
- All 16 backend tests pass, including existing Jev behavior, seeded generation,
  class balance, oracle wall bounces, simulator collisions, checkpoint metadata,
  normalized endpoint equivalence and missing-model behavior.
- Production frontend build and `git diff --check` pass.
- Headless Chrome played both actual policies through the running application.
  It confirmed unchanged canvas pixels and zero new inference requests during
  pause, switching while paused, Space shortcuts, repeated resume and one game
  animation loop. No browser errors. Screenshot: `models/browser-smoke.png`.

## Files changed

Backend:
`backend/app/main.py`, `backend/app/schemas.py`,
`backend/app/policies/{__init__,base,jev,tiny_mlp}.py`,
`backend/tests/test_training.py`.

Training:
`training/{__init__,oracle,simulator,generate_data,train,evaluate,benchmark_http}.py`,
`training/RESULTS.md`, `training/results.json`.

Frontend:
`frontend/src/components/{Game,GameCanvas}.tsx`,
`frontend/src/game/{ApiJevController,GameEngine,JevController,types}.ts`,
`frontend/src/services/jevClient.ts`, `frontend/src/styles.css`,
`frontend/tests/{api-controller.test,game.test,browser-smoke}.cjs`.

Documentation/artifacts: `README.md`, `.gitignore`; locally generated ignored
`data/training.{csv,json}` and `models/` checkpoints, logs and evaluation files.

## Next optimization

HTTP overhead now exceeds Tiny's forward time. The existing 100 ms request
interval still limits decision frequency; benchmark faster scheduling separately
before changing it. Improve imitation data with longer policy rollouts and hard
bounce scenarios, then evaluate many completed matches across speeds and
opponents. Keep Jev's current behavior for comparison. Browser inference / ONNX
has not been introduced.
