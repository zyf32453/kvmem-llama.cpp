window.G.render = function (s, canvas) {
  if (!s || !canvas) {
    return;
  }

  var ctx = canvas.getContext('2d');
  if (!ctx) {
    return;
  }

  var W = canvas.width || 0;
  var H = canvas.height || 0;
  if (W <= 0 || H <= 0) {
    return;
  }

  var p = s.player;
  if (!p) {
    return;
  }

  var wallAt = window.G.wallAt;
  if (typeof wallAt !== 'function') {
    return;
  }

  var limit = 24;
  var px = p.x;
  var py = p.y;
  var dirX = Math.cos(p.angle);
  var dirY = Math.sin(p.angle);
  var planeX = -dirY * 0.6;
  var planeY = dirX * 0.6;

  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#1b2735';
  ctx.fillRect(0, 0, W, H / 2);
  ctx.fillStyle = '#3a3a3a';
  ctx.fillRect(0, H / 2, W, H / 2);

  var zbuffer = new Float32Array(W);

  for (var x = 0; x < W; x++) {
    var t = 2 * x / W - 1;
    var rdx = dirX + planeX * t;
    var rdy = dirY + planeY * t;

    var mx = Math.floor(px);
    var my = Math.floor(py);

    var invX = rdx === 0 ? Infinity : Math.abs(1 / rdx);
    var invY = rdy === 0 ? Infinity : Math.abs(1 / rdy);

    var sideDistX = rdx < 0 ? (px - mx) * invX : (mx + 1 - px) * invX;
    var sideDistY = rdy < 0 ? (py - my) * invY : (my + 1 - py) * invY;
    if (rdx === 0) sideDistX = Infinity;
    if (rdy === 0) sideDistY = Infinity;

    var stepX = rdx < 0 ? -1 : 1;
    var stepY = rdy < 0 ? -1 : 1;
    var side = 0;
    var perp = limit;

    while (true) {
      if (sideDistX < sideDistY) {
        sideDistX += invX;
        mx += stepX;
        side = 1;
      } else {
        sideDistY += invY;
        my += stepY;
        side = 2;
      }

      if (side === 1) {
        if (sideDistX > limit) {
          break;
        }
        if (wallAt(s, mx, my)) {
          perp = sideDistX - invX;
          break;
        }
      } else {
        if (sideDistY > limit) {
          break;
        }
        if (wallAt(s, mx, my)) {
          perp = sideDistY - invY;
          break;
        }
      }
    }

    if (perp < 0.05) {
      perp = 0.05;
    }

    zbuffer[x] = perp;

    if (perp < limit) {
      var wallHeight = H / perp;
      var wallY = (H - wallHeight) / 2;
      var shade = Math.floor(Math.max(30, 200 - perp * 6));

      ctx.fillStyle = 'rgb(' + shade + ',' + Math.floor(shade * 0.8) + ',' + Math.floor(shade * 0.6) + ')';
      ctx.fillRect(x, wallY, 1, wallHeight);
    }
  }

  var bots = s.bots || [];
  var invDet = 1 / (dirX * planeY - dirY * planeX);
  var sprites = [];

  for (var i = 0; i < bots.length; i++) {
    var b = bots[i];
    if (!b || b.hp <= 0) {
      continue;
    }

    var bx = b.x - px;
    var by = b.y - py;

    var tx = invDet * (planeY * bx - planeX * by);
    var ty = invDet * (-dirY * bx + dirX * by);

    if (ty <= 0.1 || ty >= limit) {
      continue;
    }

    var screenX = Math.floor(W / 2 * (1 + tx / ty));
    if (screenX < 0 || screenX >= W) {
      continue;
    }

    if (ty >= zbuffer[screenX]) {
      continue;
    }

    var sw = H * 0.35 / ty;
    var sh = H * 0.7 / ty;
    var sx = screenX - sw / 2;
    var sy = H / 2 - sh / 2;

    sprites.push({
      x: sx,
      y: sy,
      w: sw,
      h: sh,
      depth: ty
    });
  }

  sprites.sort(function (a, b) {
    return b.depth - a.depth;
  });

  for (var j = 0; j < sprites.length; j++) {
    ctx.fillStyle = 'rgba(220,60,60,0.9)';
    ctx.fillRect(sprites[j].x, sprites[j].y, sprites[j].w, sprites[j].h);
  }

  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(W / 2 - 8, H / 2);
  ctx.lineTo(W / 2 + 8, H / 2);
  ctx.moveTo(W / 2, H / 2 - 8);
  ctx.lineTo(W / 2, H / 2 + 8);
  ctx.stroke();

  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  ctx.fillRect(8, 8, 170, 58);

  ctx.fillStyle = '#fff';
  ctx.font = '14px monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';

  ctx.fillText('HP ' + p.hp, 14, 14);
  ctx.fillText('AMMO ' + s.ammo + ' / ' + s.reserve, 14, 34);
  ctx.fillText('SCORE ' + s.score, 14, 54);

  if (s.status !== 'playing') {
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = '#fff';
    ctx.font = 'bold 48px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ctx.fillText(s.status === 'won' ? 'WON' : 'LOST', W / 2, H / 2);
  }
};