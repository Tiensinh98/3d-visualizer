import abc

__all__ = []


class GraphicsLayer:
    def __init__(self, ui_state):
        self.ui_state = ui_state
        self.render_info = None
        self.bounding_box = None

    def get_viewport(self):
        return self.ui_state.camera.viewport

    def is_constant_layer(self):
        return False

    @abc.abstractmethod
    def get_key_to_graphics_group(self):
        raise NotImplementedError
