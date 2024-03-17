import numpy as np
import qt
from . import graphics_api as ga
from . import shaders as sd
from . import gl
from . import gl_transformations as gt
from .graphics_layers import solution_layer as sl
from .graphics_objects import legend_color_bar as lcb
from . import gl_buffers as bf
from . import pick_info as pi
from . import render_info as ri

__all__ = ['GlRenderer']


class GlRenderer:
    def __init__(self, x, y, current_width, current_height, program_map, canvas, pick_info):
        self.x = x
        self.y = y
        self.current_width = current_width
        self.current_height = current_height
        self.program_map = program_map
        self.canvas = canvas
        self.ui_state = self.canvas.ui_state
        self.pick_info = pick_info
        self.color_bar = None
        self.pick_to_color = {}

    @staticmethod
    def create(x, y, current_width, current_height, context, canvas):
        shader_types = sd.SHADER_TYPES
        program_map = {}
        for shader_type in shader_types:
            program = qt.QOpenGLShaderProgram(context)
            if not program.addShaderFromSourceCode(
                    qt.QOpenGLShader.Vertex, shader_type.s_vert):
                print(program.log())
            if not program.addShaderFromSourceCode(
                    qt.QOpenGLShader.Fragment, shader_type.s_frag):
                print(program.log())
            gl.glBindAttribLocation(program.programId(), sd.VERTEX_LOCATION, 'vertex')
            for attrib_name in shader_type.attrib_names:
                gl.glBindAttribLocation(
                    program.programId(), sd.ATTRIB_INFOs[attrib_name].location_idx, attrib_name)
            program.link()
            program.bind()
            program_map[shader_type] = program
        pick_info = pi.PickInfo.create(current_width, current_height)
        return GlRenderer(x, y, current_width, current_height, program_map, canvas, pick_info)

    def free_buffers(self, width, height):
        bf.free_buffers()
        gl.glViewport(0, 0, width, height)
        if self.current_width != width or self.current_height != height:
            self.resize(width, height)

    def resize(self, width, height):
        self.current_width = width
        self.current_height = height
        self.pick_info.resize(width, height)

    def pick(self, x, y):
        obj = self.pick_info.pick(x, y, self.pick_to_color)
        return obj

    def render(self, view, graphics_scene: ga.GraphicsScene, is_render_info_dirty):
        self.free_buffers(view.width, view.height)
        gl.glClearColor(1.0, 1.0, 1.0, 0.0)
        gl.glPolygonMode(gl.GL_FRONT, gl.GL_LINE)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        graphics_layers = graphics_scene.graphics_layers
        if is_render_info_dirty:
            self.pick_to_color = {}
        for graphics_layer in graphics_layers:
            if graphics_layer.render_info is None or is_render_info_dirty:
                graphics_layer.render_info = ri.RenderInfo.create(graphics_layer, self.pick_to_color)
        solution_graphics_layer = None
        for graphics_layer in graphics_layers:
            if isinstance(graphics_layer, sl.SolutionLayer):
                solution_graphics_layer = graphics_layer
            eye_from_model = view.eye_from_model
            gl.glViewport(self.x, self.y, self.current_width, self.current_height)
            transformation = gt.GlTransformations.create(
                self.current_width, self.current_height, eye_from_model, graphics_layer.bounding_box.get_scaled(1.10))
            self.render_layer(graphics_layer, transformation)

        # render object in frame buffer for picking
        for graphics_layer in graphics_layers:
            eye_from_model = view.eye_from_model
            transformation = gt.GlTransformations.create(
                self.current_width, self.current_height, eye_from_model, graphics_layer.bounding_box.get_scaled(1.10))
            self.render_frame_buffer_layer(
                self.x, self.y, self.current_width, self.current_height, graphics_layer, transformation)
        self.render_triad_axis_name(view.height)
        field_name = solution_graphics_layer.get_field_name()
        if field_name is not None:
            self.render_color_bar(solution_graphics_layer)

    def render_layer(self, graphics_layer, transformation):
        self.render_graphics_items(sd.SCALAR_FIELD_SHADER_TYPE, transformation, graphics_layer)
        self.render_graphics_items(sd.FLAT_SHADER_TYPE, transformation, graphics_layer)

    def render_frame_buffer_layer(self, x, y, width, height, graphics_layer, transformation):
        gl.glViewport(x, y, width, height)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.pick_info.frame_buffer_id)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        self.render_graphics_items(sd.PICK_SHADER_TYPE, transformation, graphics_layer)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(x, y, width, height)

    def render_graphics_items(self, render_type, transformation, graphics_layer):
        group_graphics_render_info = graphics_layer.render_info.draw_info.get(render_type, {})
        if len(group_graphics_render_info) > 0:
            eye_from_model = transformation.eye_from_model
            scale = eye_from_model[3, 3]
            program_id = self.program_map[render_type].programId()
            gl.glUseProgram(program_id)
            for graphics_group, graphics_bodies in group_graphics_render_info.items():
                model_from_group = graphics_group.parent_from_group
                transform_type = graphics_group.transform_type
                if transform_type == ga.TransformType.Triad:
                    ndc_from_eye = transformation.get_ndc_from_eye_for_triad(-100, 100)
                    eye_from_model[0, 3] = 60 - transformation.width / 2
                    eye_from_model[1, 3] = 60 - transformation.height / 2
                    eye_from_model[2, 3] = 0
                    eye_from_model[3, 3] /= scale
                elif transform_type == ga.TransformType.FixedScale:
                    eye_from_model[3, 3] /= scale
                    ndc_from_eye = transformation.ndc_from_eye
                else:
                    ndc_from_eye = transformation.ndc_from_eye
                for graphics_body in graphics_bodies:
                    locations = GlRenderer.get_shader_locations(program_id)
                    model_from_local = np.dot(model_from_group, graphics_body.parent_from_body)
                    eye_from_local = np.ascontiguousarray(
                        np.dot(eye_from_model, model_from_local), dtype=np.float32)
                    gl.glUniformMatrix4fv(locations['ndc_from_eye'], 1, gl.GL_TRUE, ndc_from_eye)
                    gl.glUniformMatrix4fv(locations['eye_from_local'], 1, gl.GL_TRUE, eye_from_local)
                    is_discrete_colors = self.ui_state.get_is_discrete_color()
                    gl.glUniform1i(locations['is_discrete_colors'], is_discrete_colors)
                    graphics_body.render(program_id, locations)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
                gl.glUseProgram(0)

    def grab_pixels(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        pixels = gl.glReadPixels(self.x, self.y, self.current_width, self.current_height, gl.GL_RGB, gl.GL_FLOAT)
        return pixels

    def render_triad_axis_name(self, height):
        scale = GlRenderer.get_screen_scale()
        font = get_screen_font(scale)
        self.canvas.painter.begin(self.canvas)
        self.canvas.painter.beginNativePainting()
        x = 25 * scale
        y = height - 25 * scale
        color_x = [0.8, 0.2, 0.2]
        color_y = [0.2, 0.8, 0.2]
        color_z = [0.2, 0.2, 0.8]
        colors = [color_x, color_y, color_z]
        self.canvas.painter.setFont(font)
        for idx, (text, color) in enumerate(zip(list('xyz'), colors)):
            self.canvas.painter.setPen(qt.QColor.fromRgbF(*color))
            self.canvas.painter.drawText(x + (x * idx) / 2, y, text)
        self.canvas.painter.endNativePainting()
        self.canvas.painter.end()

    def render_color_bar(self, solution_graphics_layer):
        field_name = solution_graphics_layer.get_field_name()
        scale = GlRenderer.get_screen_scale()
        key_to_graphics_group = solution_graphics_layer.get_key_to_graphics_group()
        group_graphics_render_info = solution_graphics_layer.render_info.draw_info.get(sd.SCALAR_FIELD_SHADER_TYPE, {})
        solution_graphics_group = key_to_graphics_group['solutions']
        scalar_values = None
        for graphics_group, graphics_bodies in group_graphics_render_info.items():
            if graphics_group == solution_graphics_group:
                count = 0
                for graphics_body in graphics_bodies:
                    if count == 0:
                        scalar_values = graphics_body.scalar_values
                    else:
                        scalar_values = np.vstack((scalar_values, graphics_body.scalar_values))
        if scalar_values is None:
            raise ValueError("Not found solutions graphics group")
        if self.color_bar is None:
            self.color_bar = lcb.ColorBar.create(scalar_values, scale)
        else:
            self.color_bar.set_min(np.min(scalar_values))
            self.color_bar.set_max(np.max(scalar_values))
        self.color_bar.render(self.canvas, field_name)

    @staticmethod
    def get_shader_locations(program_id):
        locations = {
            'ndc_from_eye': gl.glGetUniformLocation(program_id, 'ndc_from_eye'),
            'eye_from_local': gl.glGetUniformLocation(program_id, 'eye_from_local'),
            'color': gl.glGetUniformLocation(program_id, 'color'),
            'color_range': gl.glGetUniformLocation(program_id, 'color_range'),
            'color_values': gl.glGetUniformLocation(program_id, 'color_values'),
            'min_value': gl.glGetUniformLocation(program_id, 'min_value'),
            'max_value': gl.glGetUniformLocation(program_id, 'max_value'),
            'is_discrete_colors': gl.glGetUniformLocation(program_id, 'is_discrete_colors')
        }
        return locations

    @staticmethod
    def get_screen_scale():
        default_height = 1200.
        desktop_height = qt.QApplication.desktop().screenGeometry().height()
        scale = desktop_height / default_height
        return scale


def get_screen_font(scale):
    # Use some heuristic to scale font size, I dunno.
    size = 10 * scale
    is_italic = False
    return qt.QFont("Open Sans", size, qt.QFont.Normal, is_italic)