/* CPU/offline UI validation: node tools/validate_dq_gradient_report.cjs REPORT [PNG] */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

(async () => {
  const report = path.resolve(process.argv[2]);
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.DQ_BROWSER_EXECUTABLE || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 1100 }, hasTouch: true });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route(/^https?:/, route => route.abort());
    await page.goto(pathToFileURL(report).href);
    const stack = page.locator('.gradient-curve-stack').first();
    const graphs = stack.locator(':scope > svg');
    assert.equal(await graphs.count(), 2, 'two graphs initially present');
    assert(await graphs.nth(0).isVisible() && await graphs.nth(1).isVisible());
    const bounds = await Promise.all([graphs.nth(0).boundingBox(), graphs.nth(1).boundingBox()]);
    assert.equal(bounds[0].x, bounds[1].x);
    assert.equal(bounds[0].width, bounds[1].width);
    assert(bounds[1].y >= bounds[0].y + bounds[0].height);
    const xPositions = await graphs.evaluateAll(svgs => svgs.map(svg => [...svg.querySelectorAll('.curve-hit')].map(hit => hit.getBoundingClientRect().x)));
    assert.deepEqual(xPositions[0], xPositions[1], 'identical screen-space candidate positions');
    const upper = graphs.nth(0).locator('.curve-target');
    const lower = graphs.nth(1).locator('.curve-target');
    assert((await upper.count()) > 1);
    await upper.nth(0).hover();
    assert.equal(await stack.locator('.curve-target.active').count(), 2, 'hover highlights both graphs');
    assert((await stack.locator('.curve-readout').textContent()).includes('d P50'));
    await upper.nth(0).click();
    await lower.nth(1).hover();
    assert.equal(await stack.locator('.curve-target.active[data-index="0"]').count(), 2, 'pin survives hovering another Mul');
    assert.equal(await stack.locator('.curve-target[aria-pressed="true"]').count(), 2);
    await upper.nth(0).click();
    assert.equal(await stack.locator('.curve-target[aria-pressed="true"]').count(), 0, 'click unlocks');
    await lower.nth(1).focus();
    assert.equal(await stack.locator('.curve-target.active[data-index="1"]').count(), 2, 'keyboard focus highlights both');
    await page.keyboard.press('Enter');
    assert.equal(await stack.locator('.curve-target[aria-pressed="true"]').count(), 2);
    await page.keyboard.press('Escape');
    assert.equal(await stack.locator('.curve-target.active').count(), 0);
    await lower.nth(0).focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator(':focus').getAttribute('data-index'), '1');
    await lower.nth(0).tap();
    assert.equal(await stack.locator('.curve-target.active[data-index="0"]').count(), 2, 'touch pins');
    await lower.nth(0).tap();
    assert.equal(await stack.locator('.curve-target[aria-pressed="true"]').count(), 0, 'touch unlocks');
    await upper.nth(0).focus();
    await page.keyboard.press('Enter');
    if (process.argv[3]) await stack.screenshot({ path: path.resolve(process.argv[3]) });
    await page.setViewportSize({ width: 420, height: 900 });
    assert(await graphs.nth(0).isVisible() && await graphs.nth(1).isVisible(), 'mobile still renders both graphs');
    const mobileXs = await graphs.evaluateAll(svgs => svgs.map(svg => [...svg.querySelectorAll('.curve-hit')].map(hit => hit.getBoundingClientRect().x)));
    assert.deepEqual(mobileXs[0], mobileXs[1]);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ report, checks: 'two visible graphs, aligned X, hover, pin/unpin, touch, keyboard, mobile, no JS errors', screenshot: process.argv[3] || null }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
