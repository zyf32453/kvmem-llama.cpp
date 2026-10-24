window.G.shoot = function (s) {
  if (!s || s.status !== 'playing' || s.reloadRemaining > 0 || s.ammo <= 0) {
    return false;
  }

  s.ammo -= 1;

  var p = s.player;
  var bots = s.bots || [];
  var best = null;
  var bestDist2 = Infinity;

  for (var i = 0; i < bots.length; i++) {
    var b = bots[i];
    if (!b || b.hp <= 0) {
      continue;
    }

    var dx = b.x - p.x;
    var dy = b.y - p.y;
    var angleToBot = Math.atan2(dy, dx);
    var diff = Math.atan2(Math.sin(angleToBot - p.angle), Math.cos(angleToBot - p.angle));

    if (Math.abs(diff) <= 0.12) {
      var dist2 = dx * dx + dy * dy;
      if (dist2 < bestDist2) {
        best = b;
        bestDist2 = dist2;
      }
    }
  }

  if (best) {
    var wx = best.x - p.x;
    var wy = best.y - p.y;
    var dist = Math.sqrt(bestDist2);
    var blocked = false;

    if (dist > 0) {
      var x0 = p.x;
      var y0 = p.y;
      var x1 = best.x;
      var y1 = best.y;
      var dx = x1 - x0;
      var dy = y1 - y0;

      var cellX = Math.floor(x0);
      var cellY = Math.floor(y0);
      var endCellX = Math.floor(x1);
      var endCellY = Math.floor(y1);

      if (window.G.wallAt(s, cellX, cellY) || window.G.wallAt(s, endCellX, endCellY)) {
        blocked = true;
      } else {
        var stepX = dx >= 0 ? 1 : -1;
        var stepY = dy >= 0 ? 1 : -1;
        var tDeltaX = dx !== 0 ? Math.abs(1 / dx) : Infinity;
        var tDeltaY = dy !== 0 ? Math.abs(1 / dy) : Infinity;
        var tMaxX = dx >= 0 ? (cellX + 1 - x0) * tDeltaX : (x0 - cellX) * tDeltaX;
        var tMaxY = dy >= 0 ? (cellY + 1 - y0) * tDeltaY : (y0 - cellY) * tDeltaY;

        while (true) {
          var t;

          if (tMaxX < tMaxY) {
            t = tMaxX;
            cellX += stepX;
            tMaxX += tDeltaX;
          } else {
            t = tMaxY;
            cellY += stepY;
            tMaxY += tDeltaY;
          }

          if (t >= dist) {
            break;
          }

          if (window.G.wallAt(s, cellX, cellY)) {
            blocked = true;
            break;
          }
        }
      }
    }

    if (!blocked) {
      best.hp -= s.config.damage;

      if (best.hp <= 0) {
        s.score += 1;

        var allDead = true;
        for (var k = 0; k < bots.length; k++) {
          if (bots[k] && bots[k].hp > 0) {
            allDead = false;
            break;
          }
        }

        if (allDead) {
          s.status = 'won';
        }
      }
    }
  }

  return true;
};

window.G.reload = function (s) {
  if (!s || s.status !== 'playing') {
    return false;
  }

  if (s.reloadRemaining > 0) {
    return false;
  }

  if (s.ammo >= s.config.magazine) {
    return false;
  }

  if (s.reserve <= 0) {
    return false;
  }

  s.reloadRemaining = s.config.reloadSeconds;
  return true;
};

window.G.tick = function (s, dt) {
  if (!s || s.status !== 'playing') {
    return;
  }

  if (!(dt > 0)) {
    return;
  }

  if (s.reloadRemaining <= 0) {
    return;
  }

  s.reloadRemaining -= dt;

  if (s.reloadRemaining <= 0) {
    s.reloadRemaining = 0;

    var missing = s.config.magazine - s.ammo;
    if (missing > 0 && s.reserve > 0) {
      var take = missing < s.reserve ? missing : s.reserve;
      if (take > 0) {
        s.ammo += take;
        s.reserve -= take;
      }
    }
  }
};