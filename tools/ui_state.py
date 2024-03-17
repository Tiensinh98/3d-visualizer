import model as md
import graphics as gi
import camera

__all__ = ["UiState"]

ENTITY_TYPE_TO_GRAPHICS_WINDOW_TYPE = {
    md.ComponentSystem: gi.ComponentSystemWindow,
    md.Solution: gi.SolutionWindow
}


class UiState:
    def __init__(self, model_state, application_settings):
        self.model_state = model_state
        self.application_settings = application_settings
        self.window_entity = None
        self.camera = camera.Camera()
        self.entity_to_window = {}

    @staticmethod
    def create(model_state, application_settings):
        return UiState(model_state, application_settings)

    def set_window_entity(self, entity):
        self.window_entity = entity

    def get_selected_picks(self):
        graphics_window = self.get_graphics_window()
        if graphics_window is None:
            return []
        return graphics_window.selected_picks

    def get_graphics_window(self):
        graphics_window = self.entity_to_window.get(self.window_entity, None)
        if graphics_window is not None:
            return graphics_window
        graphics_window_type = ENTITY_TYPE_TO_GRAPHICS_WINDOW_TYPE.get(self.window_entity.__class__, None)
        new_graphics_window = graphics_window_type(self, self.window_entity)
        self.entity_to_window[self.window_entity] = new_graphics_window
        return new_graphics_window

    def get_view(self):
        return self.camera.get_view()

    def get_selection_mode(self):
        return self.application_settings['selection_mode']

    def get_field_name(self):
        return self.application_settings['field_name']

    def get_is_discrete_color(self):
        return self.application_settings['is_discrete_color']
