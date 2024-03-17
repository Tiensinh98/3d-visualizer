from . import graphics_api as ga
from . import gl_renderer as gr
import view as vw

__all__ = ['GlManager', 'PaintArgs']


class GlManager:
    def __init__(self, context):
        self.context = context
        self.gl_renderer = None
        self._paint_args = PaintArgs.create_empty()

    def get_paint_args(self):
        return self._paint_args

    def set_paint_args_viewport(self, x, y, width, height):
        self._paint_args.view = self._paint_args.view.resize(x, y, width, height)

    def set_paint_args(self, paint_args):
        self._paint_args = paint_args

    def paint(self, canvas):
        paint_args = self.get_paint_args()
        view = paint_args.view
        if view.width <= 0 or view.height <= 0:
            return
        if self.gl_renderer is None:
            self.gl_renderer = gr.GlRenderer.create(
                view.x, view.y, view.width, view.height, self.context, canvas)
        graphics_scene = paint_args.graphics_scene
        is_render_info_dirty = paint_args.is_render_info_dirty
        self.gl_renderer.render(view, graphics_scene, is_render_info_dirty)

    def resize(self, width, height):
        if self.gl_renderer is not None:
            self.gl_renderer.resize(width, height)

    def grab_pixels(self):
        pixels = self.gl_renderer.grab_pixels()
        return pixels

    def delete_buffers(self):
        width, height = self._paint_args.view.width, self._paint_args.view.height
        self.gl_renderer.free_buffers(width, height)


class PaintArgs:
    def __init__(self, view, graphics_scene, is_render_info_dirty):
        self.view = view
        self.graphics_scene = graphics_scene
        self.is_render_info_dirty = is_render_info_dirty

    @staticmethod
    def create_empty():
        view = vw.View.create_default()
        graphics_scene = ga.GraphicsScene([])
        is_render_info_dirty = True
        return PaintArgs(view, graphics_scene, is_render_info_dirty)
