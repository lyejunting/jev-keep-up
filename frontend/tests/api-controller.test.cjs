const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ApiJevController } = require('./.compiled/game/ApiJevController.js');

function setup(t, fetchImplementation, options) {
  t.mock.method(globalThis, 'fetch', fetchImplementation);
  const controller = new ApiJevController(options);
  controller.start();
  t.after(() => controller.stop());
  return controller;
}

const state = { ball_x: 420, ball_y: 180, ball_vx: -2.5, ball_vy: 4.2, jev_x: 550, paddle_width: 80, game_width: 800, game_height: 600 };
const response = (prediction = 'RIGHT') => new Response(JSON.stringify({ prediction, confidence: 0.91, latency_ms: 35, provider: 'mock' }));

test('only one request at a time; sends structured JSON and respects the inference interval', async t => {
  let finish;
  let calls = 0;
  const controller = setup(t, async (url, options) => {
    calls++;
    assert.equal(url, 'http://127.0.0.1:8000/predict');
    assert.equal(options.method, 'POST');
    assert.equal(options.headers['Content-Type'], 'application/json');
    assert.deepEqual(JSON.parse(options.body), state);
    return await new Promise(resolve => { finish = resolve; });
  });
  const pending = controller.requestPrediction(state, 0);
  await Promise.resolve();
  await controller.requestPrediction(state, 200);
  assert.equal(calls, 1);
  finish(response());
  await pending;
  assert.equal(controller.decide(), 'RIGHT');
  assert.deepEqual(controller.getStatus(), { connected: true, prediction: 'RIGHT', confidence: 0.91, latencyMs: 35, provider: 'mock' });
  await controller.requestPrediction(state, 50);
  assert.equal(calls, 1);
});

test('failure clears the last decision and telemetry; subsequent requests recover', async t => {
  let failing = false;
  const controller = setup(t, async () => {
    if (failing) throw new Error('backend offline');
    return response('LEFT');
  });
  await controller.requestPrediction(state, 0);
  assert.equal(controller.decide(), 'LEFT');
  failing = true;
  await controller.requestPrediction(state, 100);
  assert.deepEqual(controller.getStatus(), { connected: false, prediction: 'STAY', confidence: null, latencyMs: null, provider: 'mock' });
  failing = false;
  await controller.requestPrediction(state, 200);
  assert.equal(controller.getStatus().connected, true);
});

test('invalid prediction and HTTP errors never drive a paddle', async t => {
  let result = response('UP');
  const controller = setup(t, async () => result);
  await controller.requestPrediction(state, 0);
  assert.equal(controller.decide(), 'STAY');
  assert.equal(controller.getStatus().connected, false);
  result = new Response('failed', { status: 500 });
  await controller.requestPrediction(state, 100);
  assert.equal(controller.decide(), 'STAY');
  assert.equal(controller.getStatus().connected, false);
});

test('timeout aborts slow requests and releases the pending slot', async t => {
  const controller = setup(t, async (_url, { signal }) => await new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
  }), { timeoutMs: 20 });
  await controller.requestPrediction(state, 0);
  assert.equal(controller.getStatus().connected, false);
  assert.equal(controller.decide(), 'STAY');
  t.mock.method(globalThis, 'fetch', async () => response());
  await controller.requestPrediction(state, 100);
  assert.equal(controller.getStatus().connected, true);
});

test('reset discards an old rally response and stop prevents further requests', async t => {
  let finish;
  let calls = 0;
  const controller = setup(t, async () => {
    calls++;
    return await new Promise(resolve => { finish = resolve; });
  });
  const pending = controller.requestPrediction(state, 0);
  await Promise.resolve();
  controller.resetPrediction();
  finish(response());
  await pending;
  assert.equal(controller.decide(), 'STAY');
  controller.stop();
  await controller.requestPrediction(state, 200);
  assert.equal(calls, 1);
});

test('continues using the last real model decision while a slower prediction is pending', async t => {
  let finish;
  let calls = 0;
  const realResponse = () => new Response(JSON.stringify({ prediction: 'LEFT', confidence: 0.82, latency_ms: 350, provider: 'openjev' }));
  const controller = setup(t, async () => {
    calls++;
    if (calls === 1) return realResponse();
    return await new Promise(resolve => { finish = resolve; });
  });
  await controller.requestPrediction(state, 0);
  const pending = controller.requestPrediction(state, 100);
  assert.equal(controller.decide(), 'LEFT');
  assert.equal(controller.getStatus().provider, 'openjev');
  await controller.requestPrediction(state, 200);
  assert.equal(calls, 2);
  finish(new Response('failed', { status: 503 }));
  await pending;
  assert.equal(controller.decide(), 'STAY');
  assert.equal(controller.getStatus().connected, false);
  assert.equal(controller.getStatus().provider, 'openjev');
});
