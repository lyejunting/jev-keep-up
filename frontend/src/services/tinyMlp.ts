import asset from '../models/keepup_mlp_v1.json';
import { normalizeState } from './jevClient';
import type { JevGameState, JevPrediction } from './jevClient';
import type { JevDecision } from '../game/types';

const actions: JevDecision[] = ['LEFT', 'STAY', 'RIGHT'];
const features = ['ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'jev_x', 'paddle_width', 'game_width', 'game_height'];
const sizes = [8, 32, 32, 3];

// Validate the committed export once, before any gameplay.
if (asset.version !== 1 || asset.normalization !== 'court-relative-v1'
    || JSON.stringify(asset.features) !== JSON.stringify(features)
    || JSON.stringify(asset.actions) !== JSON.stringify(actions)
    || JSON.stringify(asset.architecture) !== JSON.stringify(sizes)
    || asset.layers.length !== 3) throw new Error('Incompatible browser MLP');
for (const [i, layer] of asset.layers.entries()) {
  if (layer.weights.length !== sizes[i + 1] || layer.bias.length !== sizes[i + 1]
      || !layer.bias.every(Number.isFinite)
      || !layer.weights.every(row => row.length === sizes[i] && row.every(Number.isFinite))) {
    throw new Error('Invalid browser MLP weights');
  }
}

export function browserLogits(state: JevGameState): number[] {
  if (!(state.game_width > 0 && state.game_height > 0 && state.paddle_width > 0)) {
    throw new Error('Invalid game dimensions');
  }
  let values = Float32Array.from(normalizeState(state));
  if (!values.every(Number.isFinite)) throw new Error('Invalid game state');
  for (const [i, layer] of asset.layers.entries()) {
    values = Float32Array.from(layer.weights.map((row, index) => {
      let sum = layer.bias[index];
      for (let j = 0; j < row.length; j++) sum += row[j] * values[j];
      return i < 2 ? Math.max(0, sum) : sum;
    }));
  }
  if (!values.every(Number.isFinite)) throw new Error('Invalid MLP output');
  return Array.from(values);
}

export function predictTinyMlp(state: JevGameState): JevPrediction {
  const started = performance.now();
  const logits = browserLogits(state);
  const maximum = Math.max(...logits);
  const probabilities = logits.map(logit => Math.exp(logit - maximum));
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  const best = logits.indexOf(maximum); // Same first-maximum tie rule as PyTorch.
  return { prediction: actions[best], confidence: probabilities[best] / total,
    latency_ms: performance.now() - started, provider: 'tiny_mlp_browser' };
}
