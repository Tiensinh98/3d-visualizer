import numpy as np

import qt

COLOR_RANGE = np.array(
    [(0.0, 0.0, 0.7),
     (0.0, 0.0, 0.9),
     (0.0, 0.25, 1.0),
     (0.0, 0.5, 1.0),
     (0.0, 0.75, 1.0),
     (0.0, 1.0, 1.0),
     (0.25, 1.0, 0.5),
     (0.75, 1.0, 0.25),
     (1.0, 1.0, 0.0),
     (1.0, 0.75, 0.0),
     (1.0, 0.5, 0.0),
     (1.0, 0.25, 0.0),
     (0.75, 0.0, 0.0),
     (0.5, 0.0, 0.0)], dtype=np.float32)

__all__ = ['ColorBar']


class ColorBar:
    def __init__(self, min_value, max_value, scale):
        self.min_value = min_value
        self.max_value = max_value
        self.scale = scale

    @staticmethod
    def create(scalar_values, scale):
        assert isinstance(scalar_values, np.ndarray), type(scalar_values)
        mn = np.min(scalar_values)
        mx = np.max(scalar_values)
        return ColorBar(mn, mx, scale)

    def set_scale(self, scale):
        self.scale = scale

    def set_min(self, min_value):
        self.min_value = min_value

    def set_max(self, max_value):
        self.max_value = max_value

    def render(self, canvas, field_name):
        font = get_screen_font(self.scale)
        canvas.painter.begin(canvas)
        canvas.painter.beginNativePainting()
        canvas.painter.setPen(qt.Qt.black)
        canvas.painter.setFont(font)
        color_range = COLOR_RANGE[::-1]
        scalar_delta = (self.max_value - self.min_value) / len(color_range)
        width = 50 * self.scale
        height = 25 * self.scale
        origin = [width / 2, 75 * self.scale]
        color_label = f'Field: {field_name}'
        canvas.painter.drawText(origin[0], origin[1] - height, color_label)
        for idx in range(len(color_range) + 1):
            x = origin[0]
            y = origin[1] + height * idx
            # TODO: check text value of color bar
            canvas.painter.drawText(x + 1.5 * width, y, str(self.max_value - scalar_delta * idx))
            canvas.painter.drawText(x + width, y, '_')
        canvas.painter.setPen(qt.Qt.NoPen)
        for idx, color in enumerate(color_range):
            x = origin[0]
            y = origin[1] + height * idx
            canvas.painter.setBrush(qt.QColor.fromRgbF(*color))
            canvas.painter.drawRect(x, y, width, height)
        canvas.painter.endNativePainting()
        canvas.painter.end()


def get_screen_font(scale):
    # Use some heuristic to scale font size, I dunno.
    size = 10 * scale
    is_italic = False
    return qt.QFont("Open Sans", size, qt.QFont.Normal, is_italic)
