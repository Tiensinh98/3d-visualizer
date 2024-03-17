import enum

__all__ = ['TransformType', 'GraphicsScene']


class TransformType(enum.Enum):
    FixedScale = 0
    Triad = 1


class GraphicsScene:
    def __init__(self, graphics_layers):
        self.graphics_layers = graphics_layers

