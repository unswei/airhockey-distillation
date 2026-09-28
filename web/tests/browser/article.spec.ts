import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test('default article meets automated WCAG 2.1 AA checks', async ({ page }) => {
  await page.goto('./');
  const result = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
    .analyze();
  expect(result.violations).toEqual([]);
});

test('controls preserve scientific invariants', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.goto('./');
  await expect(page.locator('html')).toHaveClass('js');
  await expect(page.getByTestId('input-A')).toHaveText('(0.00, 0.00, 0)');
  await expect(page.getByTestId('input-A')).toHaveText(
    await page.getByTestId('input-B').innerText(),
  );
  await expect(page.locator('.sample-visible')).toHaveCount(10);
  await page.getByRole('button', { name: 'History emptied', exact: true }).click();
  await expect(page.locator('.sample-visible')).toHaveCount(0);
  await expect(page.locator('.observation .figure-insight')).toContainText('Ten successive masked');
  await page.getByRole('button', { name: 'Tracking returns', exact: true }).click();
  await expect(page.locator('.sample-visible')).toHaveCount(2);
  await expect(page.getByTestId('input-A')).not.toHaveText(
    await page.getByTestId('input-B').innerText(),
  );
  for (const [rank, core, total] of [
    [0, '2,304', '12,002'],
    [1, '2,467', '12,165'],
    [2, '2,630', '12,328'],
    [4, '2,956', '12,654'],
  ]) {
    await page.getByRole('button', { name: 'k = ' + rank, exact: true }).click();
    await expect(page.locator('.architecture-counts')).toContainText(core.toString());
    await expect(page.locator('.architecture-counts')).toContainText(total.toString());
    await expect(page.locator('.state-grid i')).toHaveCount(64);
    await expect(page.locator('.rank-channels i')).toHaveCount(Number(rank));
  }
  await page.getByRole('button', { name: 'Highlight new input', exact: true }).click();
  await expect(page.locator('.network-stage').first()).toHaveClass(/term-highlight/);
  const contrasts = await page.locator('.contrasts').innerText();
  await expect(page.locator('.seed-values span')).toHaveCount(35);
  await page.locator('#blackout-duration').focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('.readout-heading h4')).toHaveText('500 ms blackout');
  await expect(page.locator('.duration-title output')).toContainText('extrapolation');
  expect(await page.locator('.contrasts').innerText()).toBe(contrasts);
  await expect(page.locator('.seed-values span')).toHaveCount(35);
  await page.keyboard.press('Home');
  await expect(page.locator('.readout-heading h4')).toHaveText('0 ms blackout');
  expect(errors).toEqual([]);
});

test('assets and internal links work at the project base path', async ({ page, request }) => {
  await page.goto('./');
  for (const path of [
    'references.bib',
    'data/evidence.json',
    'data/sources.json',
    'favicon.svg',
    'kuka-setup.jpg',
  ]) {
    const response = await request.get(path);
    expect(response.status(), path).toBe(200);
    expect((await response.body()).length).toBeGreaterThan(30);
    if (path.endsWith('.json')) expect(await response.json()).toHaveProperty('schema', 1);
  }
  const anchors = await page
    .locator('a[href^="#"]')
    .evaluateAll((elements) => elements.map((a) => a.getAttribute('href')!.slice(1)));
  for (const id of anchors) await expect(page.locator('[id="' + id + '"]')).toHaveCount(1);
});

test('article header and references use plain language', async ({ page }) => {
  await page.goto('./');
  await expect(page.locator('.lab-mark')).toHaveText('UNSW SYDNEY · ROBOT LEARNING');
  await expect(page.locator('body')).not.toContainText('Reference details follow the manuscript');
  await expect(page.locator('body')).not.toContainText('Scientific fidelity');
  await expect(page.locator('a[href*="SCIENTIFIC_FIDELITY"]')).toHaveCount(0);
});

test('the paper is marked as forthcoming and its PDF is not served', async ({ page, request }) => {
  await page.goto('./');
  await expect(page.locator('.paper-pending')).toHaveText('Paper (soon)');
  await expect(page.locator('.resource-pending')).toContainText('Soon');
  await expect(page.locator('a[href*="paper.pdf"]')).toHaveCount(0);
  const response = await request.get('paper.pdf');
  // Vite may serve the article fallback for a missing path; Pages returns 404.
  expect(response.headers()['content-type']).not.toContain('application/pdf');
  expect((await response.body()).subarray(0, 5).toString()).not.toBe('%PDF-');
});

test('complete article remains readable without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({
    javaScriptEnabled: false,
    viewport: { width: 390, height: 844 },
  });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:4173/airhockey-distillation/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Linear Recurrent Memory');
  await expect(page.locator('.complete-table tbody tr')).toHaveCount(7);
  await expect(page.locator('.paper-pending')).toHaveText('Paper (soon)');
  await expect(page.locator('a[href*="paper.pdf"]')).toHaveCount(0);
  await expect(page.locator('.readout-heading h4')).toHaveText('400 ms blackout');
  await expect(page.locator('.setup-hero img')).toBeVisible();
  await expect(page.locator('.setup-hero figcaption')).toContainText('Two-robot self-play');
  await expect(page.locator('.setup-hero figcaption a')).toHaveAttribute('href', '#ref-orsula');
  await expect(page.locator('.interactive-control').first()).not.toBeVisible();
  await expect(page.locator('.print-snapshots')).toBeVisible();
  await page.getByText('All recorded means and 95% intervals', { exact: true }).click();
  await expect(page.locator('.figure-data tbody tr')).toHaveCount(8);
  await context.close();
});

for (const width of [1440, 768, 390, 320])
  test('responsive layout at ' + width + ' px', async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 960 });
    await page.goto('./');
    await page.evaluate(() => document.fonts.ready);
    expect(
      await page
        .locator('.setup-hero img')
        .evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth === 1280),
    ).toBe(true);
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true);
    for (const selector of [
      '.setup-hero',
      '.observation',
      '.architecture',
      '.performance',
      '.frontier',
      '.contrasts',
    ])
      await page
        .locator(selector)
        .screenshot({ path: testInfo.outputPath(selector.slice(1) + '.png') });
    const overflow = await page
      .locator('.figure-heading,.shot-grid,.architecture-flow,.readout-grid,.forest-row')
      .evaluateAll((elements) =>
        elements.filter((el) => el.scrollWidth > el.clientWidth + 2).map((el) => el.className),
      );
    expect(overflow).toEqual([]);
  });
