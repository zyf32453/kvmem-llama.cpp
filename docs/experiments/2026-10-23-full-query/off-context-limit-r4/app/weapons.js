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
    var rdx = best.x - p.x;
    var rdy = best.y - p.y;
    var dist = Math.sqrt(bestDist2);
    var blocked = false;

    if (dist > 0) {
      var step = 0.05;
      var samples = Math.max(1, Math.ceil(dist / step));

      for (var j = 1; j <= samples; j++) {
        var t = j / samples;
        if (window.G.wallAt(s, p.x + rdx * t, p.y + rdy * t)) {
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
    if (missing > 0) {
      var take = Math.min(missing, s.reserve);
      if (take > 0) {
        s.ammo += take;
        s.reserve -= take;
      }
    }
  }
};