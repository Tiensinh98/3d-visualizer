from . import graphics_window as gw
from .. import graphics_layers as gly

__all__ = ["SolutionWindow"]


class SolutionWindow(gw.GraphicsWindow):
    def __init__(self, ui_state, component_system):
        super().__init__(ui_state)
        self.component_system = component_system
        self.solution_layer = gly.SolutionLayer.create(ui_state, self)
        self.triad_layer = gly.TriadLayer.create(ui_state, self)
        self.selected_picks = []

    def get_graphics_layers(self):
        return [self.solution_layer, self.triad_layer]

    def get_bounding_box(self):
        return self.solution_layer.get_layer_bounding_box()
