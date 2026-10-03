const { test } = require('node:test');
const assert = require('node:assert/strict');
const { predictTinyMlp, browserLogits } = require('./.compiled/services/tinyMlp.js');
const { ApiJevController } = require('./.compiled/game/ApiJevController.js');
const references = require('./fixtures/tiny-mlp.json');

test('browser MLP reproduces exported PyTorch actions, confidence and logits', () => {
  for (const reference of references) {
    const result = predictTinyMlp(reference.state);
    assert.equal(result.prediction, reference.prediction);
    assert.ok(Math.abs(result.confidence - reference.confidence) < 2e-5);
    browserLogits(reference.state).forEach((logit, i) => assert.ok(Math.abs(logit - reference.logits[i]) < 2e-4));
    assert.equal(result.provider, 'tiny_mlp_browser');
    assert.ok(Number.isFinite(result.latency_ms) && result.latency_ms >= 0);
  }
});

test('browser MLP rejects nonfinite states and invalid dimensions', () => {
  for (const changes of [{ball_x: NaN}, {ball_vx: Infinity}, {game_width: 0}, {game_height: -1}, {paddle_width: 0}]) {
    assert.throws(() => predictTinyMlp({...references[0].state, ...changes}));
  }
});

test('browser-only controller plays and pauses without any network requests; Jev is disallowed', async t => {
  t.mock.method(globalThis, 'fetch', () => { assert.fail('Browser inference must not fetch'); });
  let predictions = 0;
  const controller = new ApiJevController({initialModel: 'tiny_mlp', allowedModels: ['tiny_mlp'],
    predictor: async state => { predictions++; return predictTinyMlp(state); }});
  controller.start();
  await controller.requestPrediction(references[0].state, 0);
  assert.equal(controller.getStatus().connected, true);
  assert.equal(controller.getStatus().provider, 'tiny_mlp_browser');
  controller.setPaused(true);
  await controller.requestPrediction(references[0].state, 200);
  assert.equal(predictions, 1);
  assert.equal(controller.decide(), 'STAY');
  assert.throws(() => controller.setModel('jev'), /unavailable/);
  controller.setPaused(false);
  await controller.requestPrediction(references[0].state, 300);
  assert.equal(predictions, 2);
  controller.stop();
});
