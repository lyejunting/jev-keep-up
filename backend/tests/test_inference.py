import unittest
import asyncio
from threading import Lock
from unittest.mock import patch
import os
import json
from types import SimpleNamespace

from fastapi import HTTPException
from pydantic import ValidationError

from app.inference import InferenceBusyError, MockJevProvider, OpenJevProvider, create_provider
from app.schemas import GameState, PredictResponse
from app import main
from app.profiling import STAGES, prediction_timings


STATE = {
    "ball_x": 420, "ball_y": 180, "ball_vx": -2.5, "ball_vy": 4.2,
    "jev_x": 550, "paddle_width": 80, "game_width": 800, "game_height": 600,
}


class InferenceTests(unittest.TestCase):
    def test_shared_prefix_is_branched_and_suffix_pooling_ignores_padding(self):
        import torch

        provider = OpenJevProvider.__new__(OpenJevProvider)
        provider.torch = torch
        provider.device = "cpu"
        provider.template = "Premise: {premise}\nHypothesis: {hypothesis}"
        provider.tokenizer = type("Tokenizer", (), {
            "pad_token_id": 0,
            "__call__": lambda self, texts, **kwargs: {"input_ids": [[10, 11, 1, 2], [10, 11, 3], [10, 11, 4, 5, 6]]},
        })()
        calls = []
        branches = []
        cache = SimpleNamespace(reorder_cache=lambda indices: branches.append(indices.tolist()))

        def backbone(**kwargs):
            self.assertFalse(torch.is_grad_enabled())
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(past_key_values=cache)
            return SimpleNamespace(last_hidden_state=kwargs["input_ids"].float().unsqueeze(-1))

        provider.backbone = backbone
        provider.model = SimpleNamespace(score=lambda pooled: torch.cat((pooled, -pooled, pooled * 0), dim=1))
        timings = dict.fromkeys(STAGES, 0.0)
        token = prediction_timings.set(timings)
        try:
            result = provider.predict_hypotheses("same premise", provider.hypotheses)
        finally:
            prediction_timings.reset(token)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["input_ids"].tolist(), [[10, 11]])
        self.assertEqual(branches, [[0, 0, 0]])
        self.assertIs(calls[1]["past_key_values"], cache)
        self.assertEqual(calls[1]["attention_mask"].tolist(), [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        expected = torch.softmax(torch.tensor([[2., -2., 0.], [3., -3., 0.], [6., -6., 0.]]), dim=-1)
        torch.testing.assert_close(result, expected)
        self.assertGreater(timings["model_forward"], 0)

    def test_http_timings_preserve_json_response_and_validation(self):
        async def request(body):
            messages = []
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            async def send(message):
                messages.append(message)
            await main.app({
                "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                "method": "POST", "scheme": "http", "path": "/predict", "raw_path": b"/predict",
                "query_string": b"", "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 8000),
            }, receive, send)
            return messages
        with patch.object(main, "provider", MockJevProvider()):
            messages = asyncio.run(request(json.dumps(STATE).encode()))
            invalid = asyncio.run(request(b'{"ball_x":'))
        self.assertEqual(messages[0]["status"], 200)
        headers = dict(messages[0]["headers"])
        timings = dict(item.split(";dur=") for item in headers[b"server-timing"].decode().split(", "))
        self.assertEqual(set(timings), set(STAGES))
        self.assertTrue(all(float(value) >= 0 for value in timings.values()))
        self.assertEqual(json.loads(messages[1]["body"])["provider"], "mock")
        self.assertEqual(invalid[0]["status"], 422)

    def test_deterministic_state_decisions(self):
        provider = MockJevProvider()
        for x, decision in [(420, "LEFT"), (550, "STAY"), (558, "STAY"), (700, "RIGHT")]:
            with self.subTest(decision=decision):
                state = GameState(**(STATE | {"ball_x": x}))
                result = provider.predict(state)
                self.assertEqual(result.prediction, decision)
                self.assertEqual(result.confidence, 0.91)
                self.assertEqual(result, provider.predict(state))

    def test_provider_selection(self):
        with patch.dict(os.environ, {}, clear=True), patch("app.inference.OpenJevProvider") as model:
            self.assertIs(create_provider(), model.return_value)
        with patch.dict(os.environ, {"JEV_PROVIDER": "mock"}):
            self.assertIsInstance(create_provider(), MockJevProvider)
        with patch("app.inference.OpenJevProvider") as model:
            self.assertIs(create_provider("openjev"), model.return_value)
        with self.assertRaises(ValueError):
            create_provider("unknown")

    def test_concurrent_model_work_is_rejected_instead_of_queued(self):
        state = GameState(**STATE)
        provider = OpenJevProvider.__new__(OpenJevProvider)
        provider._prediction_lock = Lock()
        provider._prediction_lock.acquire()
        with self.assertRaises(InferenceBusyError):
            provider.predict(state)
        with patch.object(main, "provider", provider), self.assertRaises(HTTPException) as error:
            main.predict(state)
        self.assertEqual(error.exception.status_code, 503)

    def test_startup_creates_one_provider_for_multiple_predictions(self):
        async def run():
            with patch.object(main, "create_provider", return_value=MockJevProvider()) as factory:
                async with main.lifespan(main.app):
                    for _ in range(3):
                        self.assertEqual(main.predict(GameState(**STATE)).provider, "mock")
                    factory.assert_called_once_with()
        asyncio.run(run())

    def test_failed_forward_returns_unavailable_without_mock_substitution(self):
        with patch.object(main, "provider", MockJevProvider()) as provider:
            provider.predict = lambda _state: (_ for _ in ()).throw(RuntimeError("forward failed"))
            with self.assertLogs("uvicorn.error", level="ERROR"), self.assertRaises(HTTPException) as error:
                main.predict(GameState(**STATE))
        self.assertEqual(error.exception.status_code, 503)

    def test_state_validation(self):
        for changes in [{"ball_x": float("nan")}, {"game_width": 0}, {"paddle_width": -1}, {"image": "unexpected"}]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                GameState(**(STATE | changes))

    def test_endpoint_response_has_provider(self):
        with patch.object(main, "provider", MockJevProvider()):
            result = main.predict(GameState(**STATE))
        self.assertEqual(result.prediction, "LEFT")
        self.assertEqual(result.provider, "mock")
        self.assertGreaterEqual(result.latency_ms, 0)

    def test_response_rejects_invalid_prediction_or_metrics(self):
        for changes in [{"prediction": "UP"}, {"confidence": 1.1}, {"latency_ms": -1}, {"confidence": float("nan")}]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                PredictResponse(**({"prediction": "LEFT", "confidence": 0.91, "latency_ms": 35, "provider": "mock"} | changes))


if __name__ == "__main__":
    unittest.main()
