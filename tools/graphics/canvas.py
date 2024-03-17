import qt
import graphics as gi
from . import gl as gl

__all__ = ['Canvas']


class Canvas(qt.QOpenGLWidget):
    def __init__(self, app):
        self.app = app
        self.ui_state = app.ui_state
        qt.QOpenGLWidget.__init__(self, app)
        self.gl_manager = None
        self.setMinimumSize(640, 400)
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self.width = None
        self.height = None
        self._last_pressed_mouse_x = 0
        self._last_pressed_mouse_y = 0
        self.control_pressed = False
        self.painter = qt.QPainter()
        self.setMouseTracking(True)
        self.setFocusPolicy(qt.Qt.WheelFocus)
        self.is_render_info_dirty = False

    def initializeGL(self):
        self.gl_manager = gi.GlManager(self.context())
        viewport = self.ui_state.camera.viewport
        self.gl_manager.set_paint_args_viewport(*viewport)

    def resizeGL(self, width, height):
        self.width = width
        self.height = height
        self.gl_manager.set_paint_args_viewport(0, 0, width, height)
        self.ui_state.camera.set_viewport(0, 0, width, height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.is_render_info_dirty = True

    def paintGL(self):
        if self.gl_manager is None:
            return
        graphics_window = self.ui_state.get_graphics_window()
        graphics_scene = graphics_window.get_graphics_scene()
        view = self.ui_state.get_view()
        paint_args = gi.PaintArgs(view, graphics_scene, self.is_render_info_dirty)
        self.gl_manager.set_paint_args(paint_args)
        self.gl_manager.paint(self)
        self.is_render_info_dirty = False

    def grab_pixels(self):
        self.update()
        pixels = self.gl_manager.grab_pixels()
        return pixels

    def delete_buffers(self):
        self.gl_manager.delete_buffers()

    def mouseMoveEvent(self, event):
        dx = event.x() - self._last_mouse_x
        dy = self._last_mouse_y - event.y()
        button = event.buttons()
        if button & qt.Qt.MiddleButton:
            # TODO: calculate the axis of rotation in accordance with the position of the mouse
            self.ui_state.camera.rotate(
                dx, dy, rotation_center=(0, 0, 0.))
        elif button & qt.Qt.LeftButton & self.control_pressed:
            self.ui_state.camera.pan(dx, dy)
        if button & qt.Qt.RightButton:
            menu = qt.QMenu(None)
            snap_orientation_action = qt.QAction("Snap Orientation", None)
            view_fit_action = qt.QAction("View Fit", None)
            view_fit_action.triggered.connect(self.view_fit)
            menu.addActions([snap_orientation_action, view_fit_action])
            menu.exec_(event.globalPos())
        self._last_mouse_x = event.x()
        self._last_mouse_y = event.y()
        self.update()

    @qt.Slot()
    def view_fit(self):
        graphics_window = self.ui_state.get_graphics_window()
        graphics_window.view_fit()
        self.update()

    def mousePressEvent(self, event):
        print('mousePressEvent')
        if not (event.buttons() & qt.Qt.RightButton):
            self._last_pressed_mouse_x = event.x()
            self._last_pressed_mouse_y = event.y()
            self.gl_manager.gl_renderer.pick(int(event.x()), int(event.y()))
            self.update()

    def mouseReleaseEvent(self, event):
        print('mouseReleaseEvent')
        self.update()

    def wheelEvent(self, event):
        scale_delta = event.angleDelta().y() / 2000
        scale = 1. + scale_delta
        # TODO: zoom in/out towards a certain point on the screen
        self.ui_state.camera.zoom(scale, self.width / 2, self.height / 2)
        self.update()

    def keyPressEvent(self, event):
        self.control_pressed = bool(event.modifiers() & int(qt.Qt.ControlModifier))

    def keyReleaseEvent(self, event):
        self.control_pressed = bool(event.modifiers() & int(qt.Qt.ControlModifier))
