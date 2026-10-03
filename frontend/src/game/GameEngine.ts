import { Ball } from './Ball';
import { Player } from './Player';
import type { JevController } from './JevController';
import { BALL_SPEEDS, COURT, WINNING_SCORE } from './types';
import type { BallSpeed, GameSnapshot, JevDecision, Side } from './types';

export class GameEngine {
  readonly ball = new Ball();
  readonly human = new Player('human');
  readonly jev = new Player('jev');
  private keys = new Set<string>();
  private pointTimer = 0;
  private snapshot: GameSnapshot = {
    state: 'READY', humanScore: 0, jevScore: 0, speed: 1,
    decision: 'STAY', winner: null, lastScorer: null,
  };

  constructor(private readonly controller: JevController = { decide: () => 'STAY' }) {}

  getSnapshot(): GameSnapshot { return { ...this.snapshot }; }

  start() {
    if (this.snapshot.state !== 'READY') return;
    this.ball.reset('human');
    this.snapshot.state = 'PLAYING';
  }

  restart() {
    this.snapshot = {
      state: 'READY', humanScore: 0, jevScore: 0, speed: this.snapshot.speed,
      decision: 'STAY', winner: null, lastScorer: null,
    };
    this.clearInput();
    this.human.reset();
    this.jev.reset();
    this.pointTimer = 0;
    this.start();
  }

  setSpeed(speed: BallSpeed) {
    if (BALL_SPEEDS.includes(speed)) this.snapshot.speed = speed;
  }

  setKey(key: string, pressed: boolean) {
    if (pressed) this.keys.add(key.toLowerCase());
    else this.keys.delete(key.toLowerCase());
  }

  clearInput() { this.keys.clear(); }

  update(elapsed: number) {
    if (!Number.isFinite(elapsed) || elapsed <= 0) return;
    // Small physics steps prevent the fastest ball from skipping a paddle.
    let remaining = Math.min(elapsed, 0.05);
    while (remaining > 0) {
      const dt = Math.min(remaining, 1 / 240);
      this.step(dt);
      remaining -= dt;
    }
  }

  private step(dt: number) {
    if (this.snapshot.state === 'POINT_SCORED') {
      this.pointTimer -= dt;
      if (this.pointTimer <= 0) this.snapshot.state = 'PLAYING';
      return;
    }
    if (this.snapshot.state !== 'PLAYING') return;

    const left = this.keys.has('arrowleft') || this.keys.has('a');
    const right = this.keys.has('arrowright') || this.keys.has('d');
    const humanDecision: JevDecision = left === right ? 'STAY' : left ? 'LEFT' : 'RIGHT';
    this.human.move(humanDecision, dt);

    this.snapshot.decision = this.controller.decide();
    this.jev.move(this.snapshot.decision, dt);
    this.ball.move(dt, this.snapshot.speed);
    if (this.ball.vy > 0) this.bouncePaddle(this.human);
    else this.bouncePaddle(this.jev);

    if (this.ball.y - this.ball.radius > COURT.height) this.score('jev');
    else if (this.ball.y + this.ball.radius < 0) this.score('human');
  }

  private bouncePaddle(player: Player) {
    const ball = this.ball;
    const face = player.y + (player.side === 'human' ? -player.height / 2 : player.height / 2);
    const overlapsX = Math.abs(ball.x - player.x) <= player.width / 2 + ball.radius;
    const overlapsY = Math.abs(ball.y - face) <= ball.radius;
    if (!overlapsX || !overlapsY) return;
    const offset = Math.max(-1, Math.min(1, (ball.x - player.x) / (player.width / 2)));
    const angle = offset * Math.PI / 3;
    ball.vx = Math.sin(angle) * ball.baseSpeed;
    ball.vy = Math.cos(angle) * ball.baseSpeed * (player.side === 'human' ? -1 : 1);
    ball.y = face + (player.side === 'human' ? -ball.radius : ball.radius);
  }

  private score(scorer: Side) {
    if (scorer === 'human') this.snapshot.humanScore += 1;
    else this.snapshot.jevScore += 1;
    this.snapshot.lastScorer = scorer;
    this.snapshot.decision = 'STAY';
    this.human.reset();
    this.jev.reset();
    this.ball.reset(scorer === 'human' ? 'jev' : 'human');
    if (this.snapshot.humanScore >= WINNING_SCORE || this.snapshot.jevScore >= WINNING_SCORE) {
      this.snapshot.winner = scorer;
      this.snapshot.state = 'GAME_OVER';
      this.clearInput();
    } else {
      this.snapshot.state = 'POINT_SCORED';
      this.pointTimer = 0.9;
    }
  }
}
