import type { JevDecision } from '../game/types';

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

export async function predictState(state: JevGameState, url: string, signal: AbortSignal): Promise<JevPrediction> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state),
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
