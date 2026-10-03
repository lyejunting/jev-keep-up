import os
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Protocol

from .schemas import GameState, Prediction
from .profiling import prediction_timings


class JevProvider(Protocol):
    name: str

    def predict(self, state: GameState) -> Prediction:
        ...


class MockJevProvider:
    name = "mock"

    def predict(self, state: GameState) -> Prediction:
        # Track the ball with a small dead zone to avoid oscillating in place.
        distance = state.ball_x - state.jev_x
        dead_zone = max(1, state.paddle_width * 0.15)
        decision = "STAY" if abs(distance) <= dead_zone else "LEFT" if distance < 0 else "RIGHT"
        # Fixed heuristic confidence; this is not a model probability.
        return Prediction(prediction=decision, confidence=0.91)


class OpenJevProvider:
    name = "openjev"
    repository = "AlexWortega/openjev"
    subfolder = "qwen3.5-0.8b-nli-v2s-long"
    revision = "a20448012c213128955ca0c693e7c943865cab77"
    actions = ("LEFT", "STAY", "RIGHT")
    hypotheses = (
        "The ball is left of the paddle, so moving the paddle left brings it closer to the ball.",
        "The ball is horizontally aligned with the paddle, so the paddle should stay where it is.",
        "The ball is right of the paddle, so moving the paddle right brings it closer to the ball.",
    )

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self._prediction_lock = Lock()
        self.device = os.getenv("OPENJEV_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if self.device == "cpu":
            torch.set_num_threads(min(6, os.cpu_count() or 1))
        cache_dir = os.getenv("OPENJEV_CACHE_DIR") or str(Path(__file__).resolve().parents[2] / ".cache" / "huggingface")
        started = perf_counter()
        options = dict(subfolder=self.subfolder, revision=self.revision, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repository, **options)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.repository, dtype=torch.float32, **options,
        ).to(self.device).eval()
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.template = self.model.config.nli_template
        self.entailment_id = self.model.config.label2id["entailment"]
        self.backbone = getattr(self.model, self.model.base_model_prefix)
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.load_seconds = perf_counter() - started
        logging.getLogger("uvicorn.error").info(
            "Loaded OpenJEV-style 0.8B (%s) on %s in %.2f s", self.subfolder, self.device, self.load_seconds,
        )

    @staticmethod
    def state_text(state: GameState) -> str:
        offset = state.ball_x - state.jev_x
        relation = "aligned with" if offset == 0 else f"{abs(offset):g} pixels {'left' if offset < 0 else 'right'} of"
        return (
            "Paddle game: x increases to the right. Goal: align JEV's paddle with the ball. "
            f"Court {state.game_width:g}x{state.game_height:g} pixels. "
            f"Ball center ({state.ball_x:g},{state.ball_y:g}), velocity ({state.ball_vx:g},{state.ball_vy:g}) pixels/second. "
            f"JEV paddle center x={state.jev_x:g}, width={state.paddle_width:g} pixels. "
            f"The ball is {relation} the paddle center."
        )

    def predict(self, state: GameState) -> Prediction:
        # Aborting HTTP cannot cancel a running local forward pass. Reject new
        # work while that pass completes rather than queuing it behind a lock.
        if not self._prediction_lock.acquire(blocking=False):
            raise InferenceBusyError("OpenJEV is still processing the previous prediction")
        try:
            timings = prediction_timings.get()
            started = perf_counter()
            premise = self.state_text(state)
            if timings is not None:
                timings["premise"] = (perf_counter() - started) * 1000
            torch = self.torch
            with torch.inference_mode():
                nli = self.predict_hypotheses(premise, self.hypotheses)
                started = perf_counter()
                entailment = nli[:, self.entailment_id]
                probabilities = (entailment / entailment.sum().clamp_min(1e-9)).cpu()
            # Copying to CPU synchronizes the GPU, so endpoint timing includes
            # the actual forward pass. Confidence follows the repo's decision
            # adapter: P(entailment) normalized over exactly three candidates.
            if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
                raise RuntimeError("OpenJEV returned invalid entailment probabilities")
            best = int(probabilities.argmax())
            result = Prediction(prediction=self.actions[best], confidence=float(probabilities[best]))
            if timings is not None:
                timings["post_processing"] += (perf_counter() - started) * 1000
            return result
        finally:
            self._prediction_lock.release()

    def predict_hypotheses(self, premise: str, hypotheses):
        """Repository shared-prefix inference, keeping scores on-device.

        Adapted from AlexWortega/openjev's modeling_openjev.py. Complete inputs
        are tokenized before splitting to preserve boundary token merges.
        Unlike the upstream NumPy API, the final CPU copy happens in predict()
        after entailment normalization. No model or prompt changes are needed.
        """
        torch = self.torch
        timings = prediction_timings.get()
        started = perf_counter()
        texts = [self.template.format(premise=premise.strip(), hypothesis=h.strip()) for h in hypotheses]
        sequences = self.tokenizer(texts, truncation=True, max_length=4096)["input_ids"]
        common = 0
        for column in zip(*sequences):
            if len(set(column)) != 1:
                break
            common += 1
        common = min(common, min(map(len, sequences)) - 1)
        prefix = torch.tensor([sequences[0][:common]], device=self.device)
        prefix_positions = torch.arange(common, device=self.device).unsqueeze(0)
        suffixes = [sequence[common:] for sequence in sequences]
        sizes = [len(suffix) for suffix in suffixes]
        # Compute padding width on the CPU, avoiding a GPU scalar transfer.
        width = max(sizes)
        lengths = torch.tensor(sizes, device=self.device)
        input_ids = torch.tensor(
            [suffix + [self.tokenizer.pad_token_id] * (width - len(suffix)) for suffix in suffixes],
            device=self.device,
        )
        offsets = torch.arange(width, device=self.device).unsqueeze(0)
        valid = offsets < lengths.unsqueeze(1)
        attention_mask = torch.cat((
            torch.ones((len(suffixes), common), device=self.device, dtype=torch.long), valid.long(),
        ), dim=1)
        position_ids = (common + offsets).expand(len(suffixes), -1)
        if timings is not None:
            self._synchronize()
            timings["tokenization"] = (perf_counter() - started) * 1000
        with torch.inference_mode():
            started = perf_counter()
            cache = self.backbone(
                input_ids=prefix, position_ids=prefix_positions, use_cache=True,
            ).past_key_values
            if cache is None:
                raise RuntimeError("Qwen3.5 did not return a cache for shared-prefix inference")
            # Isolate both full-attention KV and recurrent linear-attention
            # state for each action; a packed tree mask would not isolate it.
            cache.reorder_cache(torch.zeros(len(suffixes), dtype=torch.long, device=self.device))
            hidden = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=cache, use_cache=True,
            ).last_hidden_state
            pooled = hidden[torch.arange(len(suffixes), device=self.device), lengths - 1]
            logits = self.model.score(pooled).float()
            if timings is not None:
                self._synchronize()
                timings["model_forward"] = (perf_counter() - started) * 1000
            started = perf_counter()
            probabilities = torch.softmax(logits, dim=-1)
            if timings is not None:
                timings["post_processing"] = (perf_counter() - started) * 1000
            return probabilities

    def _synchronize(self):
        if self.device == "mps":
            self.torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            self.torch.cuda.synchronize()


class InferenceBusyError(RuntimeError):
    pass


def create_provider(name: str | None = None) -> JevProvider:
    selected = (name if name is not None else os.getenv("JEV_PROVIDER", "openjev")).strip().lower()
    if selected == "mock":
        return MockJevProvider()
    if selected == "openjev":
        return OpenJevProvider()
    raise ValueError(f"Unknown JEV_PROVIDER: {selected!r}; choose mock or openjev")
