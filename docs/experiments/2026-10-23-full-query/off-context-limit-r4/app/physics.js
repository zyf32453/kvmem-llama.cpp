window.G.wallAt = function (s, x, y) {
  var map = s.map;
  var ix = Math.floor(x);
  var iy = Math.floor(y);

  if (ix < 0 || iy < 0 || ix >= map.length || iy >= map.length) {
    return true;
  }

  var row = map[iy];
  if (!row || ix >= row.length) {
    return true;
  }

  return row[ix] === 1;
};

window.G.move = function (s, forward, strafe, turn, dt) {
  if (!s || s.status !== 'playing') {
    return;
  }

  var p = s.player;
  p.angle += turn * dt;

  var len = Math.sqrt(forward * forward + strafe * strafe);
  if (len > 1) {
    forward /= len;
    strafe /= len;
  }

  var speed = s.config.speed;
  var cos = Math.cos(p.angle);
  var sin = Math.sin(p.angle);

  var dx = (cos * forward - sin * strafe) * speed * dt;
  var dy = (sin * forward + cos * strafe) * speed * dt;

  if (dx === 0 && dy === 0) {
    return;
  }

  var maxStep = 0.05;
  var dist = Math.max(Math.abs(dx), Math.abs(dy));
  var steps = Math.max(1, Math.ceil(dist / maxStep));
  var sx = dx / steps;
  var sy = dy / steps;

  for (var i = 0; i < steps; i++) {
    var nx = p.x + sx;
    if (!window.G.wallAt(s, nx, p.y)) {
      p.x = nx;
    }

    var ny = p.y + sy;
    if (!window.G.wallAt(s, p.x, ny)) {
      p.y = ny;
    }
  }
};