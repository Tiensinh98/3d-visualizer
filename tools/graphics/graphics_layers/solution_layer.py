import numpy as np

from . import graphics_layer as gly
from .. import graphics_tree as gt

__all__ = ['SolutionLayer']


class SolutionLayer(gly.GraphicsLayer):
    def __init__(self, ui_state, graphics_window):
        super().__init__(ui_state)
        self.graphics_window = graphics_window
        self.component_system = graphics_window.component_system
        self.solution_graphics_group = gt.SolutionGraphicsGroup(self)
        self.edges_graphics_group = gt.EdgesGraphicsGroup(self)

    @staticmethod
    def create(ui_state, graphics_window):
        return SolutionLayer(ui_state, graphics_window)

    def get_key_to_graphics_group(self):
        key_to_graphics_group = {
            'solutions': self.solution_graphics_group,
            'edges': self.edges_graphics_group
        }
        return key_to_graphics_group

    def get_field_name(self):
        return self.ui_state.get_field_name()
