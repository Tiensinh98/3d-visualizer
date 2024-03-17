import mesh as ms

__all__ = ["Component"]


class Component:
    def __init__(self, component_jdata, component_system):
        self.component_jdata = component_jdata
        self.component_system = component_system
        mesh_filepath = component_jdata["mesh_filepath"]
        self.component_id = component_jdata["component_id"]
        self.mesh = ms.Mesh.create(mesh_filepath)

    @staticmethod
    def create(component_jdata, component_system):
        if not len(component_system.id_to_component):
            component_id = 0
        else:
            component_id = max(list(component_system.id_to_component.keys())) + 1
        component_jdata["component_id"] = component_id
        return Component(component_jdata, component_system)

    @staticmethod
    def create_json_for_insert(mesh_filepath):
        return {"mesh_filepath": mesh_filepath}

    def get_mesh_filepath(self):
        return self.component_jdata["mesh_filepath"]

    def get_field_names(self):
        return self.mesh.field_names

    def get_bounding_box(self):
        return self.mesh.get_bounding_box()
