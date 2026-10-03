import type { JevDecision } from '../game/types';

export type PolicyId = 'jev' | 'tiny_mlp';
export const POLICY_OPTIONS: { id: PolicyId; label: string }[] = [
  { id: 'jev', label: 'Jev 0.8B' }, { id: 'tiny_mlp', label: 'Tiny MLP' },
];

export interface JevGameState {
  ball_x: number;
  ball_y: number;
  ball_vx: number;
  ball_vy: number;
  jev_x: number;
  paddle_width: number;
  game_width: number;
  game_height: number;
}

export interface JevPrediction {
  prediction: JevDecision;
  confidence: number;
  latency_ms: number;
  provider: string;
}

export function normalizeState(s: JevGameState): number[] {
  return [s.ball_x / s.game_width, s.ball_y / s.game_height,
    s.ball_vx / s.game_width, s.ball_vy / s.game_height,
    s.jev_x / s.game_width, s.paddle_width / s.game_width,
    s.game_width / 960, s.game_height / 640];
}

export async function predictState(state: JevGameState, url: string, signal: AbortSignal, model: PolicyId = 'jev'): Promise<JevPrediction> {
  // Preserve the existing Jev endpoint; Tiny uses the training feature contract.
  const endpoint = model === 'jev' ? url : url.replace(/\/predict\/?$/, '/predict/normalized');
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(model === 'jev' ? state : { model, features: normalizeState(state) }),
    signal,
  });
  if (!response.ok) throw new Error(`JEV inference failed (${response.status})`);
  const data: unknown = await response.json();
  if (!data || typeof data !== 'object') throw new Error('Invalid JEV response');
  const result = data as Record<string, unknown>;
  if (!['LEFT', 'STAY', 'RIGHT'].includes(String(result.prediction))
      || typeof result.confidence !== 'number' || !Number.isFinite(result.confidence)
      || result.confidence < 0 || result.confidence > 1
      || typeof result.latency_ms !== 'number' || !Number.isFinite(result.latency_ms)
      || result.latency_ms < 0
      || typeof result.provider !== 'string' || !result.provider.trim()) {
    throw new Error('Invalid JEV prediction');
  }
  return result as unknown as JevPrediction;
}
