from .. import graphics_api as ga


class GraphicsWindow:
    def __init__(self, ui_state):
        self.ui_state = ui_state
        self.selected_picks = set()

    def get_graphics_scene(self):
        graphics_scene = ga.GraphicsScene(self.get_graphics_layers())
        return graphics_scene

    def get_graphics_layers(self):
        """
        This should return the list of layers
        """
        return []

    def get_bounding_box(self):
        raise NotImplementedError

    def get_selection_mode(self):
        return self.ui_state.get_selection_mode()

    def get_current_field_name(self):
        return self.ui_state.get_field_name()

    def view_fit(self):
        bounding_box = self.get_bounding_box()
        if bounding_box is not None and not bounding_box.is_empty():
            self.ui_state.camera.fit(bounding_box.min.ps, bounding_box.max.ps, True)


