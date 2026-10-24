G.wallAt = function (s, x, y) {
  if (!s || !s.map) return true;
  if (!isFinite(x) || !isFinite(y)) return true;

  var cx = Math.floor(x);
  var cy = Math.floor(y);
  var rows = s.map.length;
  var cols = s.map[0] ? s.map[0].length : 0;

  if (cx < 0 || cy < 0 || cx >= cols || cy >= rows) return true;
  return s.map[cy][cx] === 1;
};

G.move = function (s, forward, strafe, turn, dt) {
  if (!s || s.status !== "playing") return;

  dt = Number(dt);
  if (!isFinite(dt) || dt <= 0) return;

  var p = s.player;
  if (!p) return;

  var t = Number(turn);
  if (isFinite(t)) {
    p.angle += t * dt;
  }

  var f = Number(forward);
  var st = Number(strafe);
  if (!isFinite(f)) f = 0;
  if (!isFinite(st)) st = 0;

  var len = Math.sqrt(f * f + st * st);
  if (len === 0) return;
  if (len > 1) {
    f /= len;
    st /= len;
  }

  var speed = Number(s.config && s.config.speed);
  if (!isFinite(speed) || speed <= 0) return;

  var distance = speed * len * dt;
  if (distance <= 0) return;

  var maxStep = 0.1;
  var steps = Math.ceil(distance / maxStep);
  if (!isFinite(steps) || steps < 1) steps = 1;

  var stepDist = distance / steps;
  var cos = Math.cos(p.angle);
  var sin = Math.sin(p.angle);

  var dx = (cos * f - sin * st) * stepDist;
  var dy = (sin * f + cos * st) * stepDist;

  for (var i = 0; i < steps; i++) {
    var nx = p.x + dx;
    if (!G.wallAt(s, nx, p.y)) {
      p.x = nx;
    }

    var ny = p.y + dy;
    if (!G.wallAt(s, p.x, ny)) {
      p.y = ny;
    }
  }
};