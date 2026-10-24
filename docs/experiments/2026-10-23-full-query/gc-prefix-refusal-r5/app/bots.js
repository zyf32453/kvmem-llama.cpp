window.G.updateBots = function (s, dt) {
  if (!s || s.status !== 'playing') {
    return;
  }

  if (typeof dt !== 'number' || !isFinite(dt) || dt <= 0) {
    return;
  }

  var p = s.player;
  if (!p) {
    return;
  }

  if (!(p.hp > 0)) {
    p.hp = 0;
    s.status = 'lost';
    return;
  }

  var bots = s.bots || [];
  var moveSpeed = 1;
  var attackRange = 0.8;
  var botDamage = 10;
  var maxStep = 0.05;

  for (var i = 0; i < bots.length; i++) {
    var b = bots[i];
    if (!b || !(b.hp > 0)) {
      continue;
    }

    if (!isFinite(b.cooldown) || b.cooldown < 0) {
      b.cooldown = 0;
    }

    b.cooldown -= dt;
    if (b.cooldown < 0) {
      b.cooldown = 0;
    }

    var dx = p.x - b.x;
    var dy = p.y - b.y;
    var dist = Math.sqrt(dx * dx + dy * dy);

    if (dist > attackRange) {
      var maxMove = dist - attackRange;
      var moveDist = moveSpeed * dt;

      if (moveDist > maxMove) {
        moveDist = maxMove;
      }

      if (moveDist > 0) {
        var dirX = dx / dist;
        var dirY = dy / dist;
        var desiredX = dirX * moveDist;
        var desiredY = dirY * moveDist;

        var steps = Math.ceil(moveDist / maxStep);
        var stepX = desiredX / steps;
        var stepY = desiredY / steps;

        for (var st = 0; st < steps; st++) {
          var nx = b.x + stepX;
          if (!window.G.wallAt(s, nx, b.y)) {
            b.x = nx;
          }

          var ny = b.y + stepY;
          if (!window.G.wallAt(s, b.x, ny)) {
            b.y = ny;
          }
        }
      }

      dx = p.x - b.x;
      dy = p.y - b.y;
      dist = Math.sqrt(dx * dx + dy * dy);
    }

    if (dist <= attackRange) {
      if (b.cooldown <= 0) {
        p.hp -= botDamage;
        b.cooldown = 1;

        if (p.hp <= 0) {
          p.hp = 0;
          s.status = 'lost';
          break;
        }
      }
    }
  }
};