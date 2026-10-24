// Early failure screening only. The full 29-check browser acceptance is unchanged.
// Generated code runs in network-isolated Chromium, never in Node.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';

if (!process.argv[2] || !process.argv[3]) {
  throw new Error('Usage: node check_browser_fps_movement.mjs DRIVER_OUTPUT_DIRECTORY CHROMIUM_EXECUTABLE');
}
const root = path.resolve(process.argv[2]);
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.KVMEM_PLAYWRIGHT_MODULE || 'playwright');
const workload = JSON.parse(await fs.readFile(path.join(root, 'workload.json'), 'utf8'));
const rows = JSON.parse(await fs.readFile(path.join(root, 'results.json'), 'utf8'));
if (!['browser-fps-v1', 'browser-fps-v2'].includes(workload.name) || workload.files?.length !== 8 ||
    workload.files[0] !== 'state.js' || workload.files[1] !== 'physics.js') {
  throw new Error('Unexpected workload/file contract');
}
const report = {scope: 'first_two_turns_movement_early_failure_screen', workload: workload.name,
  fixture_revision: workload.fixture_revision || 'original', source_files: [], incomplete_turns: [],
  checks: [], browser_errors: [], blocked_requests: [],
  limitations: 'These four checks use the original first two outputs only. Passing does not replace the full 29 checks, prove later outputs, or establish GC activity, quality, or performance.'};
const assets = {};
for (let index = 0; index < 2; ++index) {
  const row = rows[index];
  const file = workload.files[index];
  if (!row?.done || row.http_status !== 200 || row.finish_reason !== 'stop' || row.error ||
      (row.output_file && row.output_file !== file)) {
    report.incomplete_turns.push(index + 1);
    continue;
  }
  const content = row.content.trim();
  // Same extraction rule as the complete checker. No code repairs or rewrites.
  const fence = content.match(/^```(?:javascript|js|html)?\s*\n([\s\S]*?)\n```$/i);
  assets[file] = fence ? fence[1] : content;
  report.source_files.push({file, turn: index + 1,
    sha256: crypto.createHash('sha256').update(assets[file]).digest('hex')});
}
if (!report.incomplete_turns.length) {
  let browser;
  try {
    browser = await chromium.launch({headless: true, executablePath: process.argv[3], args: ['--disable-gpu']});
    report.browser_version = browser.version();
    const context = await browser.newContext({serviceWorkers: 'block'});
    await context.route('**/*', route => {
      const url = new URL(route.request().url());
      const file = url.pathname.slice(1);
      if (url.origin !== 'http://gc-eval.invalid' || (file !== '' && !(file in assets))) {
        report.blocked_requests.push(url.href);
        return route.abort();
      }
      return route.fulfill({status: 200,
        body: file === '' ? '<!doctype html><script src="state.js"></script><script src="physics.js"></script>' : assets[file],
        contentType: file === '' ? 'text/html' : 'text/javascript',
        headers: {'Content-Security-Policy': "default-src 'none'; script-src 'self'; connect-src 'none'; img-src 'none'; object-src 'none'; frame-src 'none'; base-uri 'none'"}});
    });
    const page = await context.newPage();
    page.on('pageerror', error => report.browser_errors.push(error.message));
    await page.goto('http://gc-eval.invalid/', {waitUntil: 'load', timeout: 15000});
    const result = await page.evaluate(() => {
      const checks = [], measurements = {};
      const test = (name, fn) => { try { checks.push({name, pass: Boolean(fn())}); }
        catch (error) { checks.push({name, pass: false, detail: String(error)}); } };
      const make = () => G.createGame();
      // Actions and predicates are identical to check_browser_fps.mjs.
      test('out of bounds is solid', () => { const s = make(); return G.wallAt(s, -1, 2) && G.wallAt(s, 16, 2); });
      test('movement uses original speed', () => { const s = make(); G.move(s, 1, 0, 0, .1);
        measurements.straight = {final_x: s.player.x, expected_x: 2.85, tolerance: .01};
        return Math.abs(s.player.x - 2.85) < .01; });
      test('diagonal movement normalized', () => { const s = make(); G.move(s, 1, 1, 0, .1);
        measurements.diagonal = {final_x: s.player.x, final_y: s.player.y,
          actual_displacement: Math.hypot(s.player.x - 2.5, s.player.y - 2.5), expected_displacement: .35, tolerance: .01};
        return Math.abs(Math.hypot(s.player.x - 2.5, s.player.y - 2.5) - .35) < .01; });
      test('large move cannot tunnel through wall', () => { const s = make(); s.player.y = 4.5;
        G.move(s, 1, 0, 0, 1); measurements.large_step = {final_x: s.player.x, final_y: s.player.y};
        return s.player.x < 4 && !G.wallAt(s, s.player.x, s.player.y); });
      return {checks, measurements};
    });
    report.checks = result.checks;
    report.measurements = result.measurements;
  } catch (error) {
    report.execution_error = String(error);
  } finally {
    if (browser) await browser.close();
  }
}
report.passed_checks = report.checks.filter(check => check.pass).length;
report.total_checks = report.checks.length;
report.all_passed = report.incomplete_turns.length === 0 && !report.execution_error &&
  report.checks.length === 4 && report.checks.every(check => check.pass) &&
  report.browser_errors.length === 0 && report.blocked_requests.length === 0;
await fs.writeFile(path.join(root, 'movement-checks.json'), JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
console.log(JSON.stringify(report, null, 2));
process.exitCode = report.all_passed ? 0 : 1;
