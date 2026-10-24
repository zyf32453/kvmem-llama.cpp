window.G.shoot = function (s) {
  if (!s || s.status !== 'playing' || s.reloadRemaining > 0 || s.ammo <= 0) {
    return false;
  }

  s.ammo -= 1;

  var p = s.player;
  var bots = s.bots || [];
  var best = null;
  var bestDist = Infinity;
  var aimCone = 0.12;
  var twoPi = Math.PI * 2;

  for (var i = 0; i < bots.length; i++) {
    var b = bots[i];
    if (!(b.hp > 0)) {
      continue;
    }

    var dx = b.x - p.x;
    var dy = b.y - p.y;
    var dist = Math.sqrt(dx * dx + dy * dy);

    if (dist > 0) {
      var botAngle = Math.atan2(dy, dx);
      var diff = (botAngle - p.angle) % twoPi;

      if (diff > Math.PI) {
        diff -= twoPi;
      } else if (diff < -Math.PI) {
        diff += twoPi;
      }

      if (Math.abs(diff) > aimCone) {
        continue;
      }
    }

    if (dist < bestDist) {
      bestDist = dist;
      best = b;
    }
  }

  if (best) {
    var blocked = false;

    if (bestDist > 0) {
      var step = 0.01;
      var samples = Math.ceil(bestDist / step);

      for (var i2 = 1; i2 <= samples; i2++) {
        var d = i2 * step;
        if (d > bestDist) {
          d = bestDist;
        }

        var t = d / bestDist;
        var x = p.x + (best.x - p.x) * t;
        var y = p.y + (best.y - p.y) * t;

        if (window.G.wallAt(s, x, y)) {
          blocked = true;
          break;
        }
      }
    }

    if (!blocked) {
      best.hp -= s.config.damage;

      if (best.hp <= 0) {
        s.score += 1;

        var allDead = true;
        for (var j = 0; j < bots.length; j++) {
          if (bots[j].hp > 0) {
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
    return;
  }

  if (s.reloadRemaining > 0) {
    return;
  }

  if (s.ammo < s.config.magazine && s.reserve > 0) {
    s.reloadRemaining = s.config.reloadSeconds;
  }
};

window.G.tick = function (s, dt) {
  if (!s || s.status !== 'playing') {
    return;
  }

  if (typeof dt !== 'number' || !isFinite(dt) || dt <= 0) {
    return;
  }

  if (!(s.reloadRemaining > 0)) {
    return;
  }

  s.reloadRemaining -= dt;

  if (s.reloadRemaining <= 0) {
    var missing = s.config.magazine - s.ammo;
    if (missing < 0) {
      missing = 0;
    }

    if (s.reserve < 0) {
      s.reserve = 0;
    }

    if (missing > 0 && s.reserve > 0) {
      var transfer = Math.min(missing, s.reserve);
      s.ammo += transfer;
      s.reserve -= transfer;
    }

    s.reloadRemaining = 0;
  }
};