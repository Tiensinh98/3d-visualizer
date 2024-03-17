from . import component
__all__ = ["Solution"]


class Solution(component.Component):
    def __init__(self, component_jdata, component_system):
        super().__init__(component_jdata, component_system)

    @staticmethod
    def create(component_jdata, component_system):
        if not len(component_system.id_to_component):
            component_id = 0
        else:
            component_id = max(list(component_system.id_to_component.key())) + 1
        component_jdata["component_id"] = component_id
        return Solution(component_jdata, component_system)

    def get_field_names(self):
        return list(self.mesh.node_field_values.keys())

    def get_field_values_from_field_name(self, field_name):
        return self.mesh.node_field_values[field_name][0]
