import abc
import numpy as np
import bounding_box as bb


class GraphicsGroup:
    def __init__(self, graphics_layer, transform_type=None, key_to_graphics_body=None, parent_from_group=None):
        self.key_to_graphics_body = key_to_graphics_body
        self.graphics_layer = graphics_layer
        self.transform_type = transform_type
        if parent_from_group is None:
            parent_from_group = np.identity(4, dtype=np.float32)
        self.parent_from_group = parent_from_group

    @staticmethod
    def create(graphics_layer, transform_type, key_to_graphics_body, parent_from_group):
        return GraphicsGroup(graphics_layer, transform_type, key_to_graphics_body, parent_from_group)

    @abc.abstractmethod
    def get_key_to_graphics_body(self):
        if self.key_to_graphics_body is not None:
            return self.key_to_graphics_body
        return {}

