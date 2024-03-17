from ctypes import c_void_p
import copy

import numpy as np
from . import gl
from . import gl_buffers as bf
from . import shaders as sd
from . graphics_objects import tessellation as ts
import bounding_box as bb


PRIMITIVE_MODE_MAP = {1: gl.GL_POINTS, 2: gl.GL_LINES, 3: gl.GL_TRIANGLES}
COLOR_RANGE = np.ascontiguousarray(
    [(0.0, 0.0, 0.7),
     (0.0, 0.0, 0.9),
     (0.0, 0.25, 1.0),
     (0.0, 0.5, 1.0),
     (0.0, 0.75, 1.0),
     (0.0, 1.0, 1.0),
     (0.25, 1.0, 0.5),
     (0.75, 1.0, 0.25),
     (1.0, 1.0, 0.0),
     (1.0, 0.75, 0.0),
     (1.0, 0.5, 0.0),
     (1.0, 0.25, 0.0),
     (0.75, 0.0, 0.0),
     (0.5, 0.0, 0.0)], dtype=np.float32)
SHININESS = 100
NUMBER_OF_BINS = len(COLOR_RANGE)
LOW_COLOR = np.ascontiguousarray((0.0, 0.0, 0.7), dtype=np.float32)
HIGH_COLOR = np.ascontiguousarray((0.5, 0.0, 0.0), dtype=np.float32)


class GraphicsOption:
    def __init__(self, color=None, pick=None, primitive_size=None):
        if color is None:
            color = (0., 0., 0.)
        self.color = color
        if primitive_size is None:
            primitive_size = 1
        self.pick = pick
        self.primitive_size = primitive_size


class GraphicsBody:
    def __init__(self, vertices, indices, scalar_values, render_type, graphics_option, parent_from_body):
        self.bounding_box = bb.BoundingBox.create(vertices)
        attrib_values = {}
        if scalar_values is not None:
            attrib_values = {'scalar_value': scalar_values}
        self.body_buffer = bf.BodyBuffer.create(vertices, attrib_values, indices)
        primitive_mode = PRIMITIVE_MODE_MAP[self.body_buffer.primitive_mode]
        self.draw_args = (primitive_mode, self.body_buffer.indices_buffer.length,
                          self.body_buffer.indices_buffer.gl_data_type, c_void_p(0))
        self.vertices = vertices
        self.indices = indices
        self.scalar_values = scalar_values
        self.render_type = render_type
        self.graphics_option = graphics_option
        self.parent_from_body = parent_from_body

    @staticmethod
    def create(vertices, indices, render_type, scalar_values=None, parent_from_body=None, graphics_option=None):
        if parent_from_body is None:
            parent_from_body = np.identity(4, dtype=np.float32)
        return GraphicsBody(vertices, indices, scalar_values, render_type, graphics_option, parent_from_body)

    def clone(self):
        return GraphicsBody.create(self.vertices, self.indices, self.render_type,
                                   self.scalar_values, self.parent_from_body, copy.deepcopy(self.graphics_option))

    def get_bounding_box(self):
        return self.bounding_box

    def render(self, program_id, locations):
        if self.render_type in [sd.FLAT_SHADER_TYPE, sd.PICK_SHADER_TYPE]:
            if self.render_type == sd.FLAT_SHADER_TYPE:
                gl.glLineWidth(self.graphics_option.primitive_size)
            gl.glUniform3f(locations['color'], *self.graphics_option.color)
        else:
            gl.glUniform3fv(locations['color_range'], len(COLOR_RANGE), COLOR_RANGE)
            min_value = np.min(self.scalar_values)
            gl.glUniform1f(locations['min_value'], min_value)
            max_value = np.max(self.scalar_values)
            gl.glUniform1f(locations['max_value'], max_value)
            color_values = np.ascontiguousarray(np.linspace(min_value, max_value, NUMBER_OF_BINS + 1))
            gl.glUniform1fv(locations['color_values'], len(color_values), color_values)
        self.body_buffer.bind()
        gl.glDrawElements(*self.draw_args)
        self.body_buffer.unbind()


def create_generic_graphics_item(mesh, field_name, pick=None):
    vertices = mesh.get_vertices()
    scalar_values = mesh.get_scalar_values_from_field_name(field_name)
    indices = mesh.get_indices()
    render_type = sd.SCALAR_FIELD_SHADER_TYPE
    graphics_option = GraphicsOption(pick=pick)
    return GraphicsBody.create(vertices, indices, render_type,
                               scalar_values=scalar_values, graphics_option=graphics_option)


def create_edges_graphics_item(mesh):
    vertices = mesh.get_vertices()
    render_type = sd.FLAT_SHADER_TYPE
    indices = mesh.get_sedges_indices()
    return GraphicsBody.create(vertices, indices, render_type, graphics_option=GraphicsOption())


def create_sphere_graphics_item(color=None, parent_from_body=None, render_type=sd.FLAT_SHADER_TYPE):
    vertices, normals, indices = ts.tessellate_sphere()
    return GraphicsBody.create(vertices, indices, render_type,
                               parent_from_body=parent_from_body,
                               graphics_option=GraphicsOption(color=color))


def create_cylinder_graphics_item(length_ratio, color=None, parent_from_body=None,
                                  render_type=sd.FLAT_SHADER_TYPE, pick=None):
    vertices, normals, indices = ts.tessellate_cylinder(length_ratio)
    return GraphicsBody.create(vertices, indices, render_type,
                               parent_from_body=parent_from_body,
                               graphics_option=GraphicsOption(color=color, pick=pick))


def create_cone_graphics_item(radius_ratio, color=None, parent_from_body=None, render_type=sd.FLAT_SHADER_TYPE):
    vertices, normals, indices = ts.tessellate_cone(radius_ratio)
    return GraphicsBody.create(vertices, indices, render_type,
                               parent_from_body=parent_from_body, graphics_option=GraphicsOption(color=color))

