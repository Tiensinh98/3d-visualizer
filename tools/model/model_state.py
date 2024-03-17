from . import component_system as cs

__all__ = ["ModelState"]


class ModelState:
    def __init__(self, model_state_jdata):
        self.model_state_jdata = model_state_jdata
        self.component_system = cs.ComponentSystem.create(self)

    @staticmethod
    def create(model_state_jdata):
        return ModelState(model_state_jdata)

    def insert_component(self, component):
        self.component_system.insert_component(component)

    def remove_component(self, component_id):
        self.component_system.remove_component(component_id)

    def update_jdata(self, arg, json_data):
        self.model_state_jdata[arg] = json_data
