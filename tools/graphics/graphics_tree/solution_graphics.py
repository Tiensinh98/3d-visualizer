from . import graphics_group as gr
from .. import graphics_item as gi
from .. import pick as pk


class SolutionGraphicsGroup(gr.GraphicsGroup):
    def __init__(self, graphics_layer):
        super().__init__(graphics_layer)

    def get_key_to_graphics_body(self):
        component_system = self.graphics_layer.component_system
        components = component_system.get_components()
        field_name = self.graphics_layer.get_field_name()
        key_to_graphics_body = {component.component_id: gi.create_generic_graphics_item(
            component.mesh, field_name, pick=pk.Pick(component)) for component in components}
        return key_to_graphics_body
