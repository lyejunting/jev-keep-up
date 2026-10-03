// Serve frontend/dist over HTTP, then run with GAME_URL (default port 4173).
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const backendRequests = [], errors = [];
    const url = process.env.GAME_URL || 'http://127.0.0.1:4173';
    page.on('pageerror', error => errors.push(error.message));
    // Make every backend/external request fail, even if a backend happens to run.
    await page.route('**/*', route => {
      const request = route.request();
      if (new URL(request.url()).origin !== new URL(url).origin || request.url().includes('/predict')) {
        backendRequests.push(request.url()); return route.abort();
      }
      return route.continue();
    });
    await page.goto(url);
    assert.deepEqual(await page.locator('#policy option').allTextContents(), ['Tiny MLP']);
    assert.equal(await page.locator('#policy').inputValue(), 'tiny_mlp');
    await page.getByRole('button', { name: 'Start game' }).last().click();
    await page.waitForFunction(() => document.querySelector('.version')?.textContent.includes('Tiny MLP: Connected'));
    assert.equal(await page.locator('.mock-badge').textContent(), 'BROWSER');
    assert.match(await page.locator('.inference-metrics').textContent(), /\d+\.\d+ ms/);
    const playing = await page.locator('canvas').evaluate(canvas => canvas.toDataURL());
    await page.waitForTimeout(200);
    assert.notEqual(await page.locator('canvas').evaluate(canvas => canvas.toDataURL()), playing);
    await page.getByRole('button', { name: 'Pause', exact: true }).click();
    await page.waitForTimeout(150);
    const paused = await page.locator('canvas').evaluate(canvas => canvas.toDataURL());
    await page.waitForTimeout(400);
    assert.equal(await page.locator('canvas').evaluate(canvas => canvas.toDataURL()), paused);
    await page.getByRole('button', { name: 'Resume', exact: true }).last().click();
    await page.waitForTimeout(150);
    assert.notEqual(await page.locator('canvas').evaluate(canvas => canvas.toDataURL()), paused);
    await page.getByRole('button', { name: 'Restart game' }).click();
    await page.waitForFunction(() => document.querySelector('.version')?.textContent.includes('Tiny MLP: Connected'));
    assert.deepEqual(backendRequests, []);
    assert.deepEqual(errors, []);
    await page.screenshot({ path: 'models/browser-static-smoke.png', fullPage: true });
    console.log(JSON.stringify({browser_only: true, backend_requests: backendRequests.length,
      pause_resume: true, restart: true, errors}));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
