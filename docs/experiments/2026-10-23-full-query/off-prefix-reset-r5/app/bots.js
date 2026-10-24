window.G.updateBots = function (s, dt) {
  if (!s || s.status !== 'playing') {
    return;
  }

  if (!(dt > 0)) {
    return;
  }

  var p = s.player;
  var bots = s.bots || [];
  var maxStep = 0.05;

  for (var i = 0; i < bots.length; i++) {
    if (s.status !== 'playing') {
      break;
    }

    var b = bots[i];
    if (!b || b.hp <= 0) {
      continue;
    }

    b.cooldown -= dt;
    if (b.cooldown < 0) {
      b.cooldown = 0;
    }

    var dx = p.x - b.x;
    var dy = p.y - b.y;
    var dist = Math.sqrt(dx * dx + dy * dy);

    if (dist > 0.8) {
      if (dist > 0) {
        var mx = (dx / dist) * dt;
        var my = (dy / dist) * dt;

        var moveDist = Math.max(Math.abs(mx), Math.abs(my));
        var steps = Math.max(1, Math.ceil(moveDist / maxStep));
        var sx = mx / steps;
        var sy = my / steps;

        for (var j = 0; j < steps; j++) {
          var nx = b.x + sx;
          if (!window.G.wallAt(s, nx, b.y)) {
            b.x = nx;
          }

          var ny = b.y + sy;
          if (!window.G.wallAt(s, b.x, ny)) {
            b.y = ny;
          }
        }
      }
    } else if (b.cooldown <= 0) {
      b.cooldown = 1;
      p.hp -= 10;

      if (p.hp <= 0) {
        p.hp = 0;
        s.status = 'lost';
      }
    }
  }
};