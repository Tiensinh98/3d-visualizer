import numpy as np

import view
import point as pt

__all__ = ['Camera']


class Camera:
    def __init__(self):
        super().__init__()
        self.transformation_r = pt.Quaternion()
        self.transformation_t = pt.Vector()
        self.transformation_s = 1.0
        self.viewport = (0, 0, 640, 400)

    def set_viewport(self, x, y, width, height):
        self.viewport = (x, y, width, height)

    def get_view(self) -> 'view.View':
        x, y, width, height = self.viewport
        return view.View(
            x=x, y=y, width=width, height=height,
            eye_from_model=self.get_eye_from_model())

    def get_eye_from_model(self) -> pt.Transformation:
        return pt.Transformation(
            rotation=self.transformation_r,
            translation=self.transformation_t,
            scale=self.transformation_s)

    def get_eye_from_mouse(self) -> pt.Transformation:
        _, _, width, height = self.viewport
        return pt.T.translation((-width / 2.0, -height / 2.0, 0.0))

    def get_model_from_mouse(self):
        return self.get_eye_from_model().inv() * self.get_eye_from_mouse()

    def get_mouse_from_model(self):
        return self.get_eye_from_mouse().inv() * self.get_eye_from_model()

    def rotate(self, dx, dy, rotation_center=(0, 0, 0)):
        """
        Rotate the view due to a drag from pixel coordinates x1, y1 to x2, y2
        """
        if dy == 0 and dx == 0:
            return
        _, _, _, height = self.viewport
        theta = np.hypot(dx, dy) * 4.0 / height
        model_dir = self.get_model_from_mouse() * pt.Direction((-dy, dx, 0.0))
        rotation = pt.T.rotation(theta, model_dir, rotation_center)
        new_transformation = self.get_eye_from_model() * rotation
        self.transformation_r = new_transformation.r
        self.transformation_t = new_transformation.t

    def zoom(self, scale, x, y):
        _, _, width, height = self.viewport
        sx = width / 2 - x
        sy = height / 2 - y
        t = pt.T.translation((sx, sy, 0))
        s = pt.T.scaling(scale)
        t_inv = pt.T.translation((-sx, -sy, 0))  # trivial computation for 2D translation-only.
        new_transformation = t_inv * s * t * self.get_eye_from_model()
        self.transformation_t = new_transformation.t
        self.transformation_s = new_transformation.s

    def pan(self, dx, dy):
        new_transformation = pt.T.translation((dx, dy, 0)) * self.get_eye_from_model()
        # Only the translation is dirty.
        self.transformation_t = new_transformation.t

    def snap(self, axis_name: str, center):
        """
        Orient the camera to the given axis.
        """
        s = pt.T.scaling(self.transformation_s)
        r = pt.T(pt.Quaternion.from_axis_name(axis_name))
        t = pt.T.translation(center).inv()
        new_transformation = s * r * t
        self.transformation_r = new_transformation.r
        self.transformation_t = new_transformation.t
        self.transformation_s = new_transformation.s

    def fit(self, box_min, box_max, with_scaling=False):
        if with_scaling:
            radius = np.linalg.norm((box_max - box_min) / 2)
            _, _, width, height = self.viewport
            if radius < 1e-20:  # zero
                scale = 1.0
            else:
                scale = min([width, height]) / (radius * 2.0)
        else:
            scale = self.transformation_s

        s = pt.T.scaling(scale)
        r = pt.Transformation(self.transformation_r)
        t = pt.T.translation((box_min + box_max) / 2).inv()
        new_transformation = s * r * t
        self.transformation_t = new_transformation.t
        self.transformation_s = new_transformation.s

    def center(self, position):
        """
        Pan the view so that the canvas is centered at the given position,
        keeping the current scaling & rotation.
        """
        new_transformation = pt.T.translation(position).inv()
        self.transformation_t = new_transformation.t
