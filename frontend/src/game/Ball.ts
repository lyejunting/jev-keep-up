import { COURT } from './types';
import type { BallSpeed, Side } from './types';

export class Ball {
  readonly radius = 9;
  readonly baseSpeed = 330;
  x = COURT.width / 2;
  y = COURT.height / 2;
  vx = 0;
  vy = 0;

  reset(toward: Side) {
    this.x = COURT.width / 2;
    this.y = COURT.height / 2;
    const angle = (Math.random() - 0.5) * 0.8;
    this.vx = Math.sin(angle) * this.baseSpeed;
    this.vy = Math.cos(angle) * this.baseSpeed * (toward === 'human' ? 1 : -1);
  }

  move(dt: number, speed: BallSpeed) {
    this.x += this.vx * dt * speed;
    this.y += this.vy * dt * speed;
    if (this.x < this.radius) {
      this.x = 2 * this.radius - this.x;
      this.vx = Math.abs(this.vx);
    } else if (this.x > COURT.width - this.radius) {
      this.x = 2 * (COURT.width - this.radius) - this.x;
      this.vx = -Math.abs(this.vx);
    }
  }
}
