import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from pydantic import ValidationError
from fastapi import HTTPException
from app import main
from app.schemas import GameState, NormalizedRequest
from app.policies.base import normalize, FEATURES, NORMALIZATION, ACTIONS
from app.policies.tiny_mlp import TinyMLP, TinyMLPPolicy
from training.generate_data import generate
from training.oracle import OraclePolicy, target_x
from training.simulator import Simulator, play


def state(**updates):
    return GameState(**(dict(ball_x=480, ball_y=320, ball_vx=0, ball_vy=-330,
                           jev_x=480, paddle_width=128, game_width=960, game_height=640) | updates))


class TrainingTests(unittest.TestCase):
    def test_oracle_anticipates_contact_and_multiple_wall_bounces(self):
        self.assertEqual(OraclePolicy().predict(state()).prediction, 'STAY')
        self.assertEqual(OraclePolicy().predict(state(ball_vx=-330)).prediction, 'LEFT')
        self.assertEqual(OraclePolicy().predict(state(ball_vx=330)).prediction, 'RIGHT')
        self.assertAlmostEqual(target_x(state(ball_x=900, ball_y=394, ball_vx=330)), 672)
        self.assertAlmostEqual(target_x(state(ball_x=100, ball_y=394, ball_vx=330*8)), 856)

    def test_seeded_dataset_is_repeatable_balanced_and_different_across_seeds(self):
        rows, counts = generate(90, 123)
        self.assertEqual((rows, counts), generate(90, 123))
        self.assertNotEqual(rows, generate(90, 124)[0])
        self.assertEqual(counts, [30, 30, 30])
        for row in rows:
            s = GameState(**dict(zip(FEATURES, row[:8])))
            self.assertEqual(row[8], ACTIONS.index(OraclePolicy().predict(s).prediction))

    def test_simulator_collision_scoring_speed_and_seed(self):
        a, b = Simulator(42), Simulator(42)
        for _ in range(1500):
            a.step('RIGHT'); b.step('RIGHT')
        self.assertEqual(a.__dict__ | {'rng': None}, b.__dict__ | {'rng': None})
        sim = Simulator(0)
        sim.x, sim.y, sim.vx, sim.vy = 480, 575, 0, 330
        for _ in range(10): sim.step()
        self.assertLess(sim.vy, 0)
        sim.x, sim.y, sim.vx, sim.vy = 480, 65, 0, -330
        sim.step()
        self.assertGreater(sim.vy, 0)
        sim.x, sim.y, sim.vx, sim.vy = 10, 320, -330, 0
        sim.step(); sim.step()
        self.assertGreater(sim.vx, 0)
        sim.y = -20
        sim.step()
        self.assertEqual(sim.bottom_score, 1)
        self.assertEqual(sim.timer, .9)
        saved = sim.state()
        sim.step()
        self.assertEqual(sim.state(), saved)
        self.assertEqual(play(OraclePolicy(), 10, 3), play(OraclePolicy(), 10, 3))

    def test_model_parameters_checkpoint_contract_and_normalized_route(self):
        model = TinyMLP()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1443)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.pt'
            checkpoint = {'state_dict': model.state_dict(), 'features': list(FEATURES),
                          'normalization': NORMALIZATION, 'actions': list(ACTIONS)}
            torch.save(checkpoint, path)
            policy = TinyMLPPolicy(path)
            with patch.dict(main.policies, {'tiny_mlp': policy}, clear=True):
                raw = main.predict(state(), 'tiny_mlp')
                normalized = main.predict_normalized(NormalizedRequest(model='tiny_mlp', features=normalize(state())))
            self.assertEqual(raw.prediction, normalized.prediction)
            self.assertEqual(raw.confidence, normalized.confidence)
            self.assertEqual(normalized.provider, 'tiny_mlp')
            self.assertGreaterEqual(normalized.latency_ms, 0)
            checkpoint['normalization'] = 'unknown'
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError): TinyMLPPolicy(path)

    def test_missing_model_is_explicit_and_does_not_fall_back_to_jev(self):
        with patch.dict(main.policies, {}, clear=True), patch.dict(main.POLICY_FACTORIES, {'tiny_mlp': lambda: (_ for _ in ()).throw(FileNotFoundError('Train Tiny MLP first'))}):
            with self.assertRaises(HTTPException) as error:
                main.predict(state(), 'tiny_mlp')
            self.assertEqual(error.exception.status_code, 503)

    def test_normalized_features_validate_shape_dimensions_and_nonfinite(self):
        for features in [[0]*7, [0]*9, [float('nan')]*8, [0]*8]:
            with self.assertRaises(ValidationError):
                NormalizedRequest(model='tiny_mlp', features=features)


if __name__ == '__main__': unittest.main()
