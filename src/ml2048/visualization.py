from dataclasses import dataclass

BACKGROUND_COLOR = "#bbada0"


@dataclass
class TileStyle:
    color: str
    background: str
    font_size: float


def make_tile_styles(
    *,
    font_size: int = 24,
) -> list[TileStyle]:
    # color scheme and layout credit to 2048 author(s).
    spec = """
    #776e65,#cdc1b4,1,empty
    #776e65,#eee4da,1,2
    #776e65,#eee1c9,1,4
    #f9f6f2,#f3b27a,1,8
    #f9f6f2,#f69664,1,16
    #f9f6f2,#f77c5f,1,32
    #f9f6f2,#f75f3b,1,64
    #f9f6f2,#edd073,.80,128
    #f9f6f2,#edcc62,.80,256
    #f9f6f2,#edc950,.80,512
    #f9f6f2,#edc53f,.70,1024
    #f9f6f2,#edc22e,.70,2048
    #f9f6f2,#3c3a33,.70,4096
    #f9f6f2,#3c3a33,.70,8192
    #f9f6f2,#3c3a33,.55,super!
    """

    spec = spec.strip()
    styles = []
    for line in spec.splitlines():
        color, background, font_scale = line.strip().split(",")[:3]
        font_scale = float(font_scale)

        styles.append(
            TileStyle(
                color=color,
                background=background,
                font_size=font_size * font_scale,
            )
        )

    # maximum power is 16 so array length is 17
    more = 17 - len(styles)
    styles += [styles[-1]] * more

    return styles


@dataclass
class TileLoc:
    x: float
    y: float
    size: float


def make_tile_locs(
    *,
    ratio: float = 0.85,
) -> list[TileLoc]:
    """
    :param ratio: the ratio that tiles occupied in row/col
    """
    assert 0 < ratio <= 1, ratio

    tile_size = ratio / 4
    spacing_size = (1 - ratio) / 5

    res = []

    for row in range(4):
        y = spacing_size * (4 - row) + tile_size * (3 - row)
        for col in range(4):
            x = spacing_size + spacing_size * col + tile_size * col
            loc = TileLoc(
                x=x,
                y=y,
                size=tile_size,
            )
            res.append(loc)

    return res
