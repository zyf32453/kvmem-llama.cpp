window.G.render = function (s, canvas) {
  if (!s || !canvas) {
    return;
  }

  var ctx = canvas.getContext ? canvas.getContext('2d') : null;
  if (!ctx) {
    return;
  }

  var W = canvas.width || 300;
  var H = canvas.height || 150;
  if (W <= 0 || H <= 0) {
    return;
  }

  var p = s.player;
  if (!p) {
    return;
  }

  var horizon = H / 2;
  var fov = 0.66;
  var focal = (W / 2) / Math.tan(fov);
  var cosA = Math.cos(p.angle);
  var sinA = Math.sin(p.angle);

  ctx.fillStyle = '#1a1a20';
  ctx.fillRect(0, 0, W, horizon);
  ctx.fillStyle = '#2a2a30';
  ctx.fillRect(0, horizon, W, H - horizon);

  for (var x = 0; x < W; x++) {
    var cx = (x + 0.5) / W * 2 - 1;
    var rayA = p.angle + cx * fov;
    var dx = Math.cos(rayA);
    var dy = Math.sin(rayA);
    var t = 0.1;
    var hit = false;

    for (; t <= 24; t += 0.1) {
      if (window.G.wallAt(s, p.x + dx * t, p.y + dy * t)) {
        hit = true;
        break;
      }
    }

    if (hit) {
      var perp = t * Math.cos(rayA - p.angle);
      if (perp > 0.1) {
        var h = H / perp;
        var shade = Math.max(0, 1 - t / 24);
        var r = Math.floor(30 + 150 * shade);
        var g = Math.floor(30 + 150 * shade);
        var b = Math.floor(30 + 170 * shade);

        ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
        ctx.fillRect(x, horizon - h / 2, 1, h);
      }
    }
  }

  var items = [];
  var bots = s.bots || [];

  for (var i = 0; i < bots.length; i++) {
    var bot = bots[i];
    if (!bot || bot.hp <= 0) {
      continue;
    }

    var wx = bot.x - p.x;
    var wy = bot.y - p.y;
    var dist = Math.sqrt(wx * wx + wy * wy);
    var z = wx * cosA + wy * sinA;

    if (z <= 0.1 || z > 24 || dist > 24) {
      continue;
    }

    var blocked = false;
    for (var d = 0.1; d <= Math.min(dist, 24); d += 0.1) {
      if (window.G.wallAt(s, p.x + wx * (d / dist), p.y + wy * (d / dist))) {
        blocked = true;
        break;
      }
    }

    if (blocked) {
      continue;
    }

    var rx = wx * (-sinA) + wy * cosA;
    var screenX = W / 2 + (rx / z) * focal;
    var botH = (H / z) * 0.7;
    var botW = Math.max(1, botH * 0.5);
    var bottom = horizon + (H / z) * 0.5;
    var top = bottom - botH;

    if (screenX + botW / 2 < 0 || screenX - botW / 2 > W || top >= H || bottom <= 0) {
      continue;
    }

    items.push({ x: screenX, y: top, w: botW, h: botH, z: z });
  }

  items.sort(function (a, b) {
    return b.z - a.z;
  });

  ctx.fillStyle = '#e05050';
  for (var v = 0; v < items.length; v++) {
    var it = items[v];
    ctx.fillRect(it.x - it.w / 2, it.y, it.w, it.h);
  }

  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(W / 2 - 8, H / 2);
  ctx.lineTo(W / 2 + 8, H / 2);
  ctx.moveTo(W / 2, H / 2 - 8);
  ctx.lineTo(W / 2, H / 2 + 8);
  ctx.stroke();

  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(8, H - 56, 240, 48);
  ctx.fillStyle = '#ffffff';
  ctx.font = '12px monospace';
  ctx.fillText('HP ' + p.hp + '  AMMO ' + s.ammo + '/' + s.config.magazine, 14, H - 38);
  ctx.fillText('RESERVE ' + s.reserve + '  SCORE ' + s.score, 14, H - 22);

  if (s.status === 'won' || s.status === 'lost') {
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = s.status === 'won' ? '#66ff88' : '#ff6666';
    ctx.font = 'bold 48px monospace';
    ctx.fillText(s.status === 'won' ? 'WON' : 'LOST', W / 2 - 45, H / 2);
  }
};