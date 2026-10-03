import { COURT } from './types';
import type { JevDecision, Side } from './types';

export class Player {
  readonly width = 128;
  readonly height = 14;
  readonly y: number;
  readonly speed: number;
  x = COURT.width / 2;

  constructor(readonly side: Side) {
    this.y = side === 'human' ? COURT.height - 48 : 48;
    this.speed = side === 'human' ? 560 : 240;
  }

  reset() { this.x = COURT.width / 2; }

  move(decision: JevDecision, dt: number) {
    const direction = decision === 'LEFT' ? -1 : decision === 'RIGHT' ? 1 : 0;
    this.x = Math.max(this.width / 2, Math.min(
      COURT.width - this.width / 2, this.x + direction * this.speed * dt,
    ));
  }
}
