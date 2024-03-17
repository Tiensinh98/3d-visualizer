__all__ = ["ComponentSystem"]


class ComponentSystem:
    def __init__(self, model_state, component_system_jdata):
        self.model_state = model_state
        self.component_system_jdata = component_system_jdata
        self.id_to_component = {}

    @staticmethod
    def create(model_state):
        component_system_jdata = {"components": []}
        return ComponentSystem(model_state, component_system_jdata)

    def get_num_components(self):
        return len(self.id_to_component)

    def get_components(self):
        return list(self.id_to_component.values())

    def get_component(self, component_id):
        return self.id_to_component.get(component_id, None)

    def insert_component(self, component):
        self.component_system_jdata["components"].append(component.component_jdata)
        self.id_to_component[component.component_jdata['component_id']] = component

    def remove_component(self, component_id):
        idx = None
        for idx, component_jdata in enumerate(self.component_system_jdata["components"]):
            if component_jdata["component_id"] == component_id:
                idx = component_jdata["component_id"]
        if idx is None:
            return
        self.component_system_jdata["components"].remove(idx)
        self.id_to_component.pop(component_id)

