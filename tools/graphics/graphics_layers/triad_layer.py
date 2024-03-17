import numpy as np

from . import graphics_layer as gly
from .. import graphics_tree as gt
from .. import graphics_item as gi
from .. import graphics_api as ga
from .. import pick as pk
import point as pt

__all__ = ['TriadLayer']


class TriadLayer(gly.GraphicsLayer):
    def __init__(self, ui_state, graphics_window):
        super().__init__(ui_state)
        self.graphics_window = graphics_window

    @staticmethod
    def create(main, graphics_window):
        return TriadLayer(main, graphics_window)

    def get_view(self):
        return self.ui_state.get_view()

    def is_constant_layer(self):
        return True

    def get_key_to_graphics_group(self):
        color_red = (0.8, 0.2, 0.2)
        color_green = (0.2, 0.8, 0.2)
        color_blue = (0.2, 0.2, 0.8)
        parent_from_group = np.identity(4, np.float32)
        pick_x = pk.ActionPick('snap_view_x_plus')
        pick_y = pk.ActionPick('snap_view_y_plus')
        pick_z = pk.ActionPick('snap_view_z_plus')
        key_to_graphics_group = {
            'triad_group': gt.GraphicsGroup.create(
                self,
                transform_type=ga.TransformType.Triad,
                key_to_graphics_body={
                    'centre': self.create_central_sphere(),
                    'leg_x': self.create_triad_leg('+x', color_red, pick=pick_x),
                    'leg_y': self.create_triad_leg('+y', color_green, pick=pick_y),
                    'leg_z': self.create_triad_leg('+z', color_blue, pick=pick_z),
                    'tip_x': self.create_triad_tip([1, 0, 0], color_red),
                    'tip_y': self.create_triad_tip([0, 1, 0], color_green),
                    'tip_z': self.create_triad_tip([0, 0, 1], color_blue)},
                parent_from_group=parent_from_group)
        }
        return key_to_graphics_group

    def create_central_sphere(self):
        parent_from_body = pt.T.scaling(8.0).matrix()
        central_sphere = gi.create_sphere_graphics_item(color=(0.75, 0.75, 0.75), parent_from_body=parent_from_body)
        return central_sphere

    def create_triad_leg(self, axis_name, color, pick=None):
        scale = pt.T.scaling(4.0)
        if axis_name == '+z':
            transform = scale
        else:
            direction = [1, 0, 0] if axis_name == '+x' else [0, 1, 0]
            rotate = pt.T.direction_alignment((0, 0, 1), pt.Direction(direction))
            transform = rotate * scale
        return gi.create_cylinder_graphics_item(8.0, color=color, parent_from_body=transform.matrix(), pick=pick)

    def create_triad_tip(self, direction, color):
        scale = pt.T.scaling(4.0)
        translate = pt.T.translation(pt.Vector(direction) * 8.0)
        transform = scale * translate
        return gi.create_sphere_graphics_item(color=color, parent_from_body=transform.matrix())

