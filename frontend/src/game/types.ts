export type JevDecision = 'LEFT' | 'STAY' | 'RIGHT';
export type GameState = 'READY' | 'PLAYING' | 'POINT_SCORED' | 'GAME_OVER';
export type Side = 'human' | 'jev';
export const BALL_SPEEDS = [0.5, 1, 1.5, 2, 3] as const;
export type BallSpeed = (typeof BALL_SPEEDS)[number];
export const COURT = { width: 960, height: 640 } as const;
export const WINNING_SCORE = 10;

export interface GameSnapshot {
  state: GameState;
  humanScore: number;
  jevScore: number;
  speed: BallSpeed;
  decision: JevDecision;
  winner: Side | null;
  lastScorer: Side | null;
}
