import random
import numpy as np

from . import shaders as sd
import bounding_box as bb


class RenderInfo:
    def __init__(self, draw_info, pick_to_color):
        self.draw_info = draw_info
        self.pick_to_color = pick_to_color

    @staticmethod
    def create(graphics_layer, pick_to_color):
        draw_info = {}
        key_to_graphics_group = graphics_layer.get_key_to_graphics_group()
        total_bounding_box = bb.BoundingBox()
        for graphics_group in key_to_graphics_group.values():
            key_to_graphics_body = graphics_group.get_key_to_graphics_body()
            for graphics_body in key_to_graphics_body.values():
                total_bounding_box += graphics_body.get_bounding_box()
                pick = graphics_body.graphics_option.pick
                draw_info.setdefault(graphics_body.render_type, {}).setdefault(graphics_group, []).append(graphics_body)
                if pick is not None:
                    color = hash_color_pick(pick_to_color, pick)
                    pick_graphics_body = graphics_body.clone()  # clone to avoid reference
                    pick_graphics_body.graphics_option.color = color
                    pick_graphics_body.scalar_values = None
                    pick_graphics_body.render_type = sd.PICK_SHADER_TYPE
                    draw_info.setdefault(
                        sd.PICK_SHADER_TYPE, {}).setdefault(graphics_group, []).append(pick_graphics_body)
        if total_bounding_box.is_empty():
            total_bounding_box = bb.BoundingBox.create(np.zeros((2, 3)))
        graphics_layer.bounding_box = total_bounding_box
        return RenderInfo(draw_info, pick_to_color)


def hash_color_pick(pick_to_color, pick):
    idx = 2**12
    while idx in pick_to_color:
        idx = random.randint(1, 2**23)
    pick_to_color[idx] = pick
    r, g, b = encode_padic(idx)
    return r/255.0, g/255.0, b/255.0


def encode_padic(idx, b=256):
    rgb = [0, 0, 0]
    for i in range(3):
        rgb[i] = idx % b
        idx = (idx - rgb[i])/b
    assert idx == 0
    return rgb
