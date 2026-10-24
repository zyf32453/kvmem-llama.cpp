window.G = {
  createGame: function () {
    var map = [];

    for (var y = 0; y < 16; y++) {
      var row = [];
      for (var x = 0; x < 16; x++) {
        row.push(0);
      }
      map.push(row);
    }

    for (var i = 0; i < 16; i++) {
      map[0][i] = 1;
      map[15][i] = 1;
      map[i][0] = 1;
      map[i][15] = 1;
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
      status: "playing",
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