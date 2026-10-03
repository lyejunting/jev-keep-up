import type { GameEngine } from './GameEngine';
import { COURT } from './types';

export function renderGame(ctx: CanvasRenderingContext2D, engine: GameEngine) {
  const { width, height } = COURT;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#141c19';
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = '#21312a';
  ctx.lineWidth = 1;
  for (let x = 40; x < width; x += 40) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
  }
  for (let y = 40; y < height; y += 40) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  ctx.strokeStyle = '#3b4b40';
  ctx.setLineDash([8, 12]);
  ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(width / 2, height / 2, 64, 0, Math.PI * 2); ctx.stroke();

  ctx.font = '12px monospace';
  ctx.textAlign = 'left';
  ctx.fillStyle = '#708275';
  ctx.fillText('JEV', 24, 30);
  ctx.fillText('HUMAN / YOU', 24, height - 22);

  for (const player of [engine.jev, engine.human]) {
    ctx.fillStyle = player.side === 'human' ? '#c3f66b' : '#bca4f8';
    ctx.shadowColor = ctx.fillStyle;
    ctx.shadowBlur = 16;
    ctx.beginPath();
    ctx.roundRect(player.x - player.width / 2, player.y - player.height / 2, player.width, player.height, 7);
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  const { ball } = engine;
  ctx.fillStyle = '#f5f8ed';
  ctx.shadowColor = '#e3ffbf';
  ctx.shadowBlur = 18;
  ctx.beginPath(); ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2); ctx.fill();
  ctx.shadowBlur = 0;
}
