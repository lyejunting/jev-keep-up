# JEV Keep Up

A paddle game against AI. Move your paddle, keep the ball in play, and reach 10 points to win.

[Play the browser version](https://glittering-otter-06d482.netlify.app/).

## Models

| Where you play | Available models | What you need |
| --- | --- | --- |
| Public browser version | Tiny MLP | A browser |
| Local development | Jev 0.8B and Tiny MLP | Node.js and the Python backend |

Tiny MLP runs inside the browser. Jev runs through the local Python backend.

## Controls

- **Move:** Left/Right arrows or A/D. Touch controls are also available.
- **Pause or resume:** the button or Space.
- **Restart:** starts a fresh match.
- **Ball speed:** can be changed during play.

## Run locally

Use Node.js 20.19+ or 22.12+, Python 3.10+, and Make. Run these commands from the repository root.

### First-time setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
npm --prefix frontend install
```

### Start the game

```sh
make start
```

Open **http://127.0.0.1:5173/**. Both models are available in the dropdown.

The first start downloads Jev. The launcher frees ports 8000 and 5173 before starting both services. Press Ctrl+C to stop them, or run `make stop`.

To play Tiny MLP locally without starting Python:

```sh
npm --prefix frontend run dev
```

Select **Tiny MLP**; Jev requires the backend.

## Upload to Netlify

Install the frontend dependencies if you have not already, then build:

```sh
npm --prefix frontend install
npm --prefix frontend run build:static
```

**Upload the entire `frontend/dist/` folder to Netlify Drop.** Include both `index.html` and `assets/`.

This build offers only Tiny MLP and makes no backend requests. Players do not need Python or an API key. See [Netlify's upload instructions](https://docs.netlify.com/deploy/create-deploys/#drag-and-drop).

To preview the static build locally:

```sh
npm --prefix frontend run preview
```

For deployment from Git, `netlify.toml` already specifies the frontend build and publish directory.

## Train Tiny MLP

With the Python environment installed, generate examples, train, and evaluate:

```sh
.venv/bin/python training/generate_data.py --samples 30000 --seed 42
.venv/bin/python training/train.py --epochs 100
.venv/bin/python training/evaluate.py --samples 3000 --games 100 --simulate-latency
```

Training uses actions from a deterministic oracle. The best checkpoint is saved to `models/keepup_mlp_v1.pt`. Generated datasets and Python checkpoints are ignored by Git.

### Update the browser model after training

```sh
.venv/bin/python training/export_browser.py
npm --prefix frontend run test:compile
.venv/bin/python training/verify_browser.py --fixtures frontend/tests/fixtures/tiny-mlp.json
npm --prefix frontend test
npm --prefix frontend run build:static
```

The export updates `frontend/src/models/keepup_mlp_v1.json`, which is bundled with the game. Upload the rebuilt `frontend/dist/` folder to publish the new model.

## Checks and results

```sh
npm --prefix frontend test
PYTHONPATH=backend:. .venv/bin/python -m unittest discover -s backend/tests
```

See [training/RESULTS.md](training/RESULTS.md) for accuracy, gameplay success, and latency measurements.
