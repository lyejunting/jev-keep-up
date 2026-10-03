import { predictState } from '../services/jevClient';
import type { JevGameState, PolicyId } from '../services/jevClient';
import type { JevController } from './JevController';
import type { JevDecision } from './types';

export interface JevInferenceStatus {
  connected: boolean;
  prediction: JevDecision;
  confidence: number | null;
  latencyMs: number | null;
  provider: string | null;
}

interface Options {
  url?: string;
  intervalMs?: number;
  timeoutMs?: number;
}

const unavailable = (provider: string | null = 'openjev'): JevInferenceStatus => ({
  connected: false, prediction: 'STAY', confidence: null, latencyMs: null, provider,
});

export class ApiJevController implements JevController {
  private readonly url: string;
  private readonly intervalMs: number;
  private readonly timeoutMs: number;
  private status = unavailable();
  private active = false;
  private paused = false;
  private model: PolicyId = 'jev';
  private inFlight = false;
  private lastInference = -Infinity;
  private generation = 0;
  private request: AbortController | null = null;

  constructor({ url = 'http://127.0.0.1:8000/predict', intervalMs = 100, timeoutMs = 30000 }: Options = {}) {
    this.url = url;
    this.intervalMs = Number.isFinite(intervalMs) && intervalMs >= 50 ? intervalMs : 100;
    this.timeoutMs = Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : 30000;
  }

  decide(): JevDecision { return this.status.prediction; }
  getStatus(): JevInferenceStatus { return { ...this.status }; }

  getModel(): PolicyId { return this.model; }

  setModel(model: PolicyId) {
    if (model === this.model) return;
    this.resetPrediction();
    this.model = model;
    this.status = unavailable(model === 'jev' ? 'openjev' : 'tiny_mlp');
    this.lastInference = -Infinity;
  }

  setPaused(paused: boolean) {
    if (this.paused === paused) return;
    this.paused = paused;
    this.resetPrediction();
    this.lastInference = -Infinity;
  }

  start() {
    this.active = true;
    this.lastInference = -Infinity;
  }

  stop() {
    this.active = false;
    this.resetPrediction();
    this.status = unavailable(this.status.provider);
  }

  resetPrediction() {
    // Discard responses from a previous rally, restart, or component mount.
    this.generation += 1;
    this.request?.abort();
    this.status = { ...this.status, prediction: 'STAY', confidence: null, latencyMs: null };
  }

  async requestPrediction(state: JevGameState, time: number): Promise<void> {
    if (!this.active || this.paused || this.inFlight || time - this.lastInference < this.intervalMs) return;
    this.lastInference = time;
    this.inFlight = true;
    const generation = this.generation;
    const request = new AbortController();
    this.request = request;
    const timeout = setTimeout(() => request.abort(), this.timeoutMs);

    try {
      const result = await predictState(state, this.url, request.signal, this.model);
      if (!this.active || this.paused || generation !== this.generation) return;
      this.status = {
        connected: true, prediction: result.prediction,
        confidence: result.confidence, latencyMs: result.latency_ms, provider: result.provider,
      };
    } catch {
      if (this.active && generation === this.generation) this.status = unavailable(this.status.provider);
    } finally {
      clearTimeout(timeout);
      if (this.request === request) this.request = null;
      this.inFlight = false;
    }
  }
}
