// Optional end-to-end check. Start both services, install playwright, then run
// node frontend/tests/browser-smoke.cjs (CHROME_PATH can select installed Chrome).
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    let requests = 0;
    const errors = [];
    page.on('request', request => { if (request.url().includes('/predict')) requests++; });
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(() => {
      const request = window.requestAnimationFrame.bind(window);
      const cancel = window.cancelAnimationFrame.bind(window);
      const pending = new Set();
      window.__maxPendingFrames = 0;
      window.requestAnimationFrame = callback => {
        const isGameLoop = callback.toString().includes('pixelWidth');
        const id = request(time => { pending.delete(id); callback(time); });
        if (isGameLoop) pending.add(id);
        window.__maxPendingFrames = Math.max(window.__maxPendingFrames, pending.size);
        return id;
      };
      window.cancelAnimationFrame = id => { pending.delete(id); cancel(id); };
    });
    await page.goto(process.env.GAME_URL || 'http://127.0.0.1:5173');
    await page.getByRole('button', { name: 'Start game' }).last().click();
    const waitConnected = async label => {
      await page.waitForFunction(label => document.querySelector('.version')?.textContent.includes(`${label}: Connected`), label, { timeout: 60000 });
    };
    await waitConnected('Jev 0.8B');
    await page.getByRole('button', { name: 'Pause', exact: true }).click();
    await page.waitForTimeout(150);
    const canvas = await page.locator('canvas').evaluate(canvas => canvas.toDataURL());
    const pausedRequests = requests;
    await page.keyboard.press('ArrowLeft');
    await page.waitForTimeout(800);
    assert.equal(await page.locator('canvas').evaluate(canvas => canvas.toDataURL()), canvas);
    assert.equal(requests, pausedRequests, 'paused UI must issue no inference requests');
    await page.selectOption('#policy', 'tiny_mlp');
    await page.waitForTimeout(300);
    assert.equal(requests, pausedRequests, 'switching model while paused must not infer');
    await page.getByRole('button', { name: 'Resume', exact: true }).last().click();
    await waitConnected('Tiny MLP');
    const tinyRequests = requests;
    await page.waitForTimeout(300);
    assert.equal(requests, tinyRequests, 'Tiny MLP runs locally without API requests');
    for (let i = 0; i < 5; i++) {
      await page.getByRole('button', { name: 'Pause', exact: true }).click();
      await page.getByRole('button', { name: 'Resume', exact: true }).last().click();
    }
    await page.locator('h1').click(); // move focus away from buttons/select
    await page.keyboard.press('Space');
    await page.waitForFunction(() => document.querySelector('.match-state')?.textContent === 'PAUSED');
    await page.keyboard.press('Space');
    await waitConnected('Tiny MLP');
    await page.selectOption('#policy', 'jev');
    await waitConnected('Jev 0.8B');
    assert.equal(await page.evaluate(() => window.__maxPendingFrames), 1, 'exactly one animation loop');
    assert.deepEqual(errors, []);
    await page.screenshot({ path: process.env.SCREENSHOT_PATH || 'models/browser-smoke.png', fullPage: true });
    console.log(JSON.stringify({ real_jev: true, tiny_mlp: true, pause_freezes_canvas_and_requests: true,
      switch_while_paused: true, space_shortcut: true, animation_loops: 1, requests, errors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
