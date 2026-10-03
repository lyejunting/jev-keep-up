const { test } = require('node:test');
const assert = require('node:assert/strict');
const { GameEngine } = require('./.compiled/game/GameEngine.js');

function game() {
  const engine = new GameEngine({ decide: () => 'STAY' });
  engine.start();
  return engine;
}

function miss(engine, side) {
  engine.ball.x = 20;
  engine.ball.y = side === 'human' ? -20 : 660;
  engine.ball.vx = 0;
  engine.ball.vy = side === 'human' ? -330 : 330;
  engine.update(1 / 240);
}

test('ready is stationary; start launches one ball and repeated start keeps the match', () => {
  const engine = new GameEngine();
  engine.update(0.05);
  assert.equal(engine.getSnapshot().state, 'READY');
  assert.equal(engine.ball.y, 320);
  engine.start();
  engine.update(0.05);
  assert.equal(engine.getSnapshot().state, 'PLAYING');
  assert.ok(engine.ball.y > 320);
  const y = engine.ball.y;
  engine.start();
  assert.equal(engine.ball.y, y);
});

test('each speed applies immediately to the current rally without resetting the ball', () => {
  const engine = game();
  engine.ball.vx = 0;
  engine.ball.vy = 330;
  for (const speed of [0.5, 1, 1.5, 2, 3]) {
    const before = engine.ball.y;
    engine.setSpeed(speed);
    assert.equal(engine.ball.y, before);
    engine.update(0.01);
    assert.ok(Math.abs(engine.ball.y - before - 330 * 0.01 * speed) < 1e-8);
    assert.equal(engine.getSnapshot().speed, speed);
  }
});

test('both paddles return a fast ball, and side walls reflect it', () => {
  const engine = game();
  engine.setSpeed(3);
  engine.ball.x = 480;
  engine.ball.y = 575;
  engine.ball.vx = 0;
  engine.ball.vy = 330;
  engine.update(0.03);
  assert.ok(engine.ball.vy < 0);
  engine.ball.y = 65;
  engine.update(0.03);
  assert.ok(engine.ball.vy > 0);
  engine.ball.x = 10;
  engine.ball.vx = -330;
  engine.ball.vy = 0;
  engine.update(0.01);
  assert.ok(engine.ball.vx > 0);
  assert.ok(engine.ball.x >= engine.ball.radius);
});

test('a miss awards exactly one opponent point, centers the ball, then resumes', () => {
  const engine = game();
  miss(engine, 'jev');
  assert.equal(engine.getSnapshot().jevScore, 1);
  assert.equal(engine.getSnapshot().humanScore, 0);
  assert.equal(engine.getSnapshot().state, 'POINT_SCORED');
  assert.equal(engine.ball.x, 480);
  assert.equal(engine.ball.y, 320);
  engine.update(0.05);
  assert.equal(engine.ball.y, 320);
  for (let i = 0; i < 20; i++) engine.update(0.05);
  assert.equal(engine.getSnapshot().state, 'PLAYING');
  assert.equal(engine.getSnapshot().jevScore, 1);
  miss(engine, 'human');
  assert.equal(engine.getSnapshot().humanScore, 1);
});

test('first to ten freezes the match; restart clears scores and preserves chosen speed', () => {
  for (const winner of ['human', 'jev']) {
    const engine = game();
    engine.setSpeed(2);
    for (let point = 0; point < 10; point++) {
      miss(engine, winner);
      if (point < 9) {
        for (let i = 0; i < 19; i++) engine.update(0.05);
      }
    }
    assert.equal(engine.getSnapshot().state, 'GAME_OVER');
    assert.equal(engine.getSnapshot().winner, winner);
    assert.equal(engine.getSnapshot()[winner === 'human' ? 'humanScore' : 'jevScore'], 10);
    const y = engine.ball.y;
    engine.update(0.05);
    assert.equal(engine.ball.y, y);
    engine.restart();
    assert.equal(engine.getSnapshot().state, 'PLAYING');
    assert.equal(engine.getSnapshot().humanScore, 0);
    assert.equal(engine.getSnapshot().jevScore, 0);
    assert.equal(engine.getSnapshot().speed, 2);
    assert.equal(engine.getSnapshot().winner, null);
  }
});

test('keyboard aliases move and clamp the human paddle; releasing or clearing stops it', () => {
  const engine = game();
  engine.setKey('ArrowLeft', true);
  engine.update(0.05);
  assert.ok(engine.human.x < 480);
  engine.setKey('ArrowLeft', false);
  const x = engine.human.x;
  engine.update(0.05);
  assert.equal(engine.human.x, x);
  engine.setKey('D', true);
  for (let i = 0; i < 30; i++) engine.update(0.05);
  assert.ok(engine.human.x <= 960 - engine.human.width / 2);
  engine.clearInput();
  const stopped = engine.human.x;
  engine.update(0.05);
  assert.equal(engine.human.x, stopped);
});

test('engine consumes cached decisions without passing game coordinates', () => {
  let decision = 'RIGHT';
  const controller = { decide(...args) { assert.equal(args.length, 0); return decision; } };
  const engine = new GameEngine(controller);
  engine.start();
  engine.ball.x = 800;
  engine.update(0.01);
  assert.equal(engine.getSnapshot().decision, 'RIGHT');
  assert.ok(engine.jev.x > 480);
  decision = 'STAY';
  const x = engine.jev.x;
  engine.update(0.01);
  assert.equal(engine.getSnapshot().decision, 'STAY');
  assert.equal(engine.jev.x, x);
});
