window.G = {
  createGame: function () {
    var map = [];

    for (var y = 0; y < 16; y++) {
      var row = [];
      for (var x = 0; x < 16; x++) {
        row.push((x === 0 || x === 15 || y === 0 || y === 15) ? 1 : 0);
      }
      map.push(row);
    }

    map[4][4] = 1;

    return {
      map: map,
      player: {
        x: 2.5,
        y: 2.5,
        angle: 0,
        hp: 100
      },
      bots: [
        {
          x: 10.5,
          y: 8.5,
          hp: 40,
          cooldown: 0
        }
      ],
      ammo: 7,
      reserve: 21,
      reloadRemaining: 0,
      status: 'playing',
      score: 0,
      config: {
        magazine: 7,
        reloadSeconds: 1.25,
        damage: 20,
        speed: 3.5
      }
    };
  }
};