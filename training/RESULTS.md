# Keep-Up results

Measured locally on **3 October 2026**.

The public game uses **Tiny MLP in the browser**. Local development also supports **Jev 0.8B through Python**.

## Tiny MLP training and holdout

The model has **1,443 parameters**: 8 inputs, two 32-neuron hidden layers with ReLU, and 3 action outputs.

We generated **30,000 examples**, balanced equally across LEFT, STAY, and RIGHT. A deterministic oracle supplies the labels. Training used Adam and CrossEntropyLoss for 100 epochs; the best checkpoint was from epoch 93.

| Dataset | Samples | Accuracy |
| --- | ---: | ---: |
| Training — 80% | 24,000 | 88.92% |
| Validation holdout — 20% | 6,000 | 88.07% |
| Independent test | 3,000 | 89.37% |

Validation selects the best checkpoint. The independent test contains separately generated states unused during training or checkpoint selection.

## Jev versus Tiny MLP success

| Measure | Jev 0.8B | Tiny MLP |
| --- | ---: | ---: |
| Oracle action agreement | 46.7% — 14/30 | 80.0% — 24/30 |
| Paddle return success | 66.7% — 4/6 | 100.0% — 6/6 |
| Completed-match win rate | Not available | Not available |

**Oracle agreement** measures how often the model chooses the oracle's action. Both models were tested on the same 30 states.

**Paddle return success** measures hits divided by hits plus misses. Each model played a separate 20-second simulation against the same oracle opponent, starting from the same randomized opening seed. Decisions included each model's measured median inference delay.

This is a small comparison: only six paddle encounters per model. Neither match finished, so these results do not establish a match win rate.

### Larger Tiny MLP gameplay test

Across **100 randomized-opening simulations**, each capped at 120 seconds:

| Metric | Tiny MLP |
| --- | ---: |
| Paddle return success | 99.33% |
| Average paddle returns | 35.74 |
| Average score | 0.12 |
| Median / best score | 0 / 1 |
| Completed matches | 0 |

The opponent was the faster bottom-paddle oracle. Most play continued as rallies rather than completed matches. Jev has not been tested on this larger set, so it is not a direct comparison with Jev's figures above.

## Latency

Both backend policies were measured on **100 identical states**, after five warmup requests per model. Jev ran on Apple Metal; the Python Tiny MLP ran on CPU.

| Model | Inference median | Inference p95 | HTTP median | HTTP p95 |
| --- | ---: | ---: | ---: | ---: |
| Jev 0.8B | 207.00 ms | 214.63 ms | 208.53 ms | 216.15 ms |
| Tiny MLP — Python | 0.04 ms | 0.06 ms | 0.66 ms | 0.98 ms |

**These are backend measurements, not browser latency measurements.** Inference excludes model loading and HTTP transport. HTTP timing includes the local request and response. Neither includes the game's 100 ms decision interval.

To repeat the benchmark with the backend running:

```sh
.venv/bin/python training/benchmark_http.py --samples 100
```

## Browser version

The exported MLP weights occupy approximately **30 KB of JSON** and are bundled with the frontend. Gameplay requires no backend requests.

On **3,000 independent states**, browser inference chose the same action as PyTorch **100% of the time**. Its oracle agreement remained **89.37%**.

Browser latency is displayed in the game; a desktop and mobile latency benchmark is still needed.

## Verification and next step

The last implementation checks passed:

- **22 frontend tests** and **16 backend tests**.
- Static and local production builds.
- Browser-only gameplay, pause/resume, and restart with **zero backend requests**.
- Local switching between real Jev and browser Tiny MLP, including switching while paused.

The next useful evaluation is a larger comparison using the same game seeds for both models, with completed wins, losses, and timeouts reported separately.

Detailed measurements are in [results.json](results.json). Setup and static upload instructions are in the [README](../README.md).
