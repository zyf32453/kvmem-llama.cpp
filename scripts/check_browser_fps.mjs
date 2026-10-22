// Execute model-generated browser files only inside Chromium, without network
// access or Node bindings. This is a bounded functional case study, not an agent.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const root = path.resolve(process.argv[2] || '');
if (!process.argv[2]) throw new Error('Usage: node check_browser_fps.mjs RESULTS_DIRECTORY [CHROMIUM_EXECUTABLE]');
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.KVMEM_PLAYWRIGHT_MODULE || 'playwright');
const workload = JSON.parse(await fs.readFile(path.join(root, 'workload.json'), 'utf8'));
const rows = JSON.parse(await fs.readFile(path.join(root, 'results.json'), 'utf8'));
const allowed = ['state.js', 'physics.js', 'weapons.js', 'bots.js', 'render.js', 'index.html'];
if (!['browser-fps-v1', 'browser-fps-v2'].includes(workload.name) || workload.files?.length !== 8 ||
    workload.files.some(file => !allowed.includes(file))) throw new Error('Unexpected workload/file contract');
const assets = {};
const incomplete = [];
for (let i = 0; i < Math.max(workload.files.length, rows.length); ++i) {
  const row = rows[i];
  const file = row?.output_file || workload.files[i];
  if (file && !allowed.includes(file)) throw new Error('Unexpected output file');
  if (!row?.done || row.http_status !== 200 || row.finish_reason !== 'stop' || row.error) {
    incomplete.push(i + 1);
    continue;
  }
  const content = row.content.trim();
  const fence = content.match(/^```(?:javascript|js|html)?\s*\n([\s\S]*?)\n```$/i);
  if (!file) throw new Error('Repair output lacks file identity');
  assets[file] = fence ? fence[1] : content;
}
const app = path.join(root, 'app');
await fs.mkdir(app); // Never silently replace a previous inspection.
for (const [file, text] of Object.entries(assets)) await fs.writeFile(path.join(app, file), text);
const report = { workload: workload.name, incomplete_turns: incomplete, checks: [], browser_errors: [],
  blocked_requests: [], limitations: 'Deterministic browser checks only; no visual-playability score or general task-success claim.' };
function check(name, pass, detail) { report.checks.push({ name, pass: Boolean(pass), ...(detail ? { detail } : {}) }); }
check('all requested outputs completed without truncation', incomplete.length === 0);
check('all six application files exist', allowed.every(file => file in assets));
if (allowed.every(file => file in assets)) {
  const browser = await chromium.launch({ headless: true, ...(process.argv[3] ? { executablePath: process.argv[3] } : {}) });
  try {
    const context = await browser.newContext({ viewport: { width: 1000, height: 700 }, serviceWorkers: 'block' });
    await context.route('**/*', route => {
      const url = new URL(route.request().url());
      const file = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
      const isolated = file === 'state-only.html';
      if (url.origin !== 'http://gc-eval.invalid' || (!allowed.includes(file) && !isolated)) {
        report.blocked_requests.push(url.href);
        return route.abort();
      }
      return route.fulfill({ status: 200, body: isolated ? '<!doctype html><script src="state.js"></script>' : assets[file],
        contentType: file.endsWith('.html') ? 'text/html' : 'text/javascript',
        headers: { 'Content-Security-Policy': "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'" } });
    });
    const page = await context.newPage();
    page.on('pageerror', error => report.browser_errors.push(error.message));
    await page.goto('http://gc-eval.invalid/', { waitUntil: 'load', timeout: 15000 });
    await page.waitForTimeout(150);
    report.public_functions = await page.evaluate(() => Object.fromEntries(
      ['createGame', 'wallAt', 'move', 'shoot', 'reload', 'tick', 'updateBots', 'render']
        .map(name => [name, typeof window.G?.[name]])));
    const functional = await page.evaluate(() => {
      const checks = [];
      const test = (name, fn) => { try { checks.push({ name, pass: Boolean(fn()) }); }
        catch (e) { checks.push({ name, pass: false, detail: String(e) }); } };
      const make = () => G.createGame();
      test('required public functions', () => ['createGame', 'wallAt', 'move', 'shoot', 'reload', 'tick', 'updateBots', 'render']
        .every(name => typeof G[name] === 'function'));
      test('original tuning and final round field retained', () => {
        const s = make(), c = s.config;
        return s.player.x === 2.5 && s.player.y === 2.5 && s.player.angle === 0 && s.player.hp === 100 &&
          s.ammo === 7 && s.reserve === 21 && s.reloadRemaining === 0 && s.status === 'playing' && s.score === 0 &&
          c.magazine === 7 && c.reloadSeconds === 1.25 && c.damage === 20 && c.speed === 3.5 && s.round === 1 &&
          s.bots.length === 1 && s.bots[0].x === 10.5 && s.bots[0].y === 8.5 && s.bots[0].hp === 40;
      });
      test('original map and boundary walls retained', () => {
        const s = make(); return s.map.length === 16 && s.map.every(r => r.length === 16) && s.map[4][4] === 1 &&
          s.map[0].every(v => v === 1) && s.map[15].every(v => v === 1) && s.map.every(r => r[0] === 1 && r[15] === 1);
      });
      test('restart states share no mutable data', () => {
        const a = make(), b = make(); a.map[0][0] = 0; a.player.hp = 0; a.bots[0].hp = 0; a.config.speed = 0;
        return b.map[0][0] === 1 && b.player.hp === 100 && b.bots[0].hp === 40 && b.config.speed === 3.5;
      });
      test('out of bounds is solid', () => { const s = make(); return G.wallAt(s, -1, 2) && G.wallAt(s, 16, 2); });
      test('movement uses original speed', () => { const s = make(); G.move(s, 1, 0, 0, .1); return Math.abs(s.player.x - 2.85) < .01; });
      test('diagonal movement normalized', () => { const s = make(); G.move(s, 1, 1, 0, .1);
        return Math.abs(Math.hypot(s.player.x - 2.5, s.player.y - 2.5) - .35) < .01; });
      test('large move cannot tunnel through wall', () => { const s = make(); s.player.y = 4.5;
        G.move(s, 1, 0, 0, 1); return s.player.x < 4 && !G.wallAt(s, s.player.x, s.player.y); });
      test('miss consumes one bullet', () => { const s = make(); return G.shoot(s) === true && s.ammo === 6; });
      test('empty magazine never auto reloads', () => { const s = make(); s.ammo = 0;
        const shot = G.shoot(s); G.tick(s, 5); return shot === false && s.ammo === 0 && s.reserve === 21; });
      test('reload cannot restart and uses original duration', () => { const s = make(); s.ammo = 0; G.reload(s);
        if (Math.abs(s.reloadRemaining - 1.25) > 1e-6) return false;
        G.tick(s, .25); const left = s.reloadRemaining; G.reload(s); return Math.abs(left - 1) < 1e-6 && s.reloadRemaining === left; });
      test('cannot fire while reloading', () => { const s = make(); s.ammo = 3; G.reload(s);
        return G.shoot(s) === false && s.ammo === 3; });
      test('large reload tick transfers once', () => { const s = make(); s.ammo = 0; G.reload(s); G.tick(s, 5); G.tick(s, 5);
        return s.ammo === 7 && s.reserve === 14 && s.reloadRemaining === 0; });
      test('partial reserve cannot become negative', () => { const s = make(); s.ammo = 4; s.reserve = 2; G.reload(s); G.tick(s, 5);
        return s.ammo === 6 && s.reserve === 0; });
      test('wall blocks damage', () => { const s = make(); s.player.y = 4.5; s.bots = [{ x: 6.5, y: 4.5, hp: 40, cooldown: 0 }];
        G.shoot(s); return s.bots[0].hp === 40; });
      test('nearest target hit, original damage and no dead-bot rescoring', () => { const s = make();
        s.bots = [{ x: 4.5, y: 2.5, hp: 40, cooldown: 0 }, { x: 4.5, y: 3.5, hp: 40, cooldown: 0 }];
        G.shoot(s); if (s.bots[0].hp !== 20) return false; G.shoot(s); G.shoot(s);
        return s.bots[0].hp <= 0 && s.score === 1 && s.bots[1].hp === 40; });
      test('last kill wins and stops movement/shooting', () => { const s = make(); s.bots[0].x = 4.5; s.bots[0].y = 2.5;
        G.shoot(s); G.shoot(s); const x = s.player.x, ammo = s.ammo; G.move(s, 1, 0, 0, .1);
        return s.status === 'won' && s.score === 1 && G.shoot(s) === false && s.ammo === ammo && s.player.x === x; });
      test('bot attack can end round with zero HP', () => { const s = make(); s.player.hp = 10; s.bots[0].x = 2.6; s.bots[0].y = 2.5;
        G.updateBots(s, .1); return s.player.hp === 0 && s.status === 'lost'; });
      test('bot attack cooldown', () => { const s = make(); s.bots[0].x = 2.6; s.bots[0].y = 2.5;
        G.updateBots(s, .1); const hp = s.player.hp; G.updateBots(s, .1); return hp === 90 && s.player.hp === hp; });
      test('render does not mutate state and draws scene', () => { const s = make(), before = JSON.stringify(s);
        const c = document.createElement('canvas'); c.width = 320; c.height = 200; G.render(s, c);
        const pixels = c.getContext('2d').getImageData(0, 0, 320, 200).data;
        return before === JSON.stringify(s) && pixels.some((v, i) => i % 4 !== 3 && v !== 0); });
      return checks;
    });
    report.checks.push(...functional);
    check('page has game canvas', await page.locator('canvas#game').count() === 1);
    if (await page.locator('canvas#game').count() === 1 && await page.evaluate(() => typeof window.G?.createGame === 'function' && !!window.game)) {
      // Mutate the exposed current state in place: the contract does not require
      // input handlers to observe reassignment of window.game from a debugger.
      await page.evaluate(() => { Object.assign(window.game, G.createGame()); });
      const x = await page.evaluate(() => window.game.player.x);
      await page.keyboard.down('w'); await page.waitForTimeout(150); await page.keyboard.up('w');
      check('W key drives animation loop movement', await page.evaluate(old => window.game.player.x > old, x));
      await page.evaluate(() => { Object.assign(window.game, G.createGame()); window.game.ammo = 0; });
      await page.keyboard.press('r'); await page.waitForTimeout(40);
      check('R key starts manual reload', await page.evaluate(() => window.game.reloadRemaining > 0));
      await page.evaluate(() => { Object.assign(window.game, G.createGame()); window.game.status = 'won'; window.game.player.hp = 5; });
      await page.keyboard.press('Enter'); await page.waitForTimeout(40);
      check('Enter restarts a finished round', await page.evaluate(() => window.game.status === 'playing' && window.game.player.hp === 100 && window.game.ammo === 7));
      await page.locator('canvas#game').click({ position: { x: 400, y: 225 } });
      check('canvas click fires', await page.evaluate(() => window.game.ammo === 6));
    } else {
      for (const name of ['W key drives animation loop movement', 'R key starts manual reload',
        'Enter restarts a finished round', 'canvas click fires']) check(name, false, 'Application did not initialize; interaction cannot be tested.');
    }
    await page.screenshot({ path: path.join(root, 'app.png'), fullPage: true });
    check('no uncaught browser errors', report.browser_errors.length === 0);
    check('no unexpected network/dependency requests', report.blocked_requests.length === 0);
    // A separate diagnostic, not a repaired application or a replacement for
    // the failed integration checks: can the final file recall the old values?
    const statePage = await context.newPage();
    await statePage.goto('http://gc-eval.invalid/state-only.html', { waitUntil: 'load', timeout: 15000 });
    report.isolated_final_state = await statePage.evaluate(() => {
      try {
        const s = G.createGame(), c = s.config || {};
        const expected = { ammo: 7, reserve: 21, reloadSeconds: 1.25, damage: 20, speed: 3.5,
          magazine: 7, health: 100, botHealth: 40, round: 1 };
        const actual = { ammo: s.ammo, reserve: s.reserve, reloadSeconds: c.reloadSeconds,
          damage: c.damage, speed: c.speed, magazine: c.magazine, health: s.player?.hp,
          botHealth: s.bots?.[0]?.hp, round: s.round };
        return { expected, actual, retained: Object.keys(expected).filter(k => actual[k] === expected[k]),
          missing_or_changed: Object.keys(expected).filter(k => actual[k] !== expected[k]) };
      } catch (error) { return { error: String(error) }; }
    });
  } catch (error) {
    check('browser inspection completed', false, String(error));
  } finally { await browser.close(); }
}
report.passed = report.checks.filter(c => c.pass).length;
report.total = report.checks.length;
report.all_passed = report.passed === report.total;
await fs.writeFile(path.join(root, 'browser-checks.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
process.exitCode = report.all_passed ? 0 : 1;
