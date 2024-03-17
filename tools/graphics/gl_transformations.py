import numpy as np
import point as pt


class GlTransformations:
    __slots__ = ['ndc_from_eye', 'ndc_from_model', 'eye_from_model',
                 'eye_from_mouse', 'near', 'far', 'width', 'height']

    def __init__(self, ndc_from_eye, ndc_from_model, eye_from_model, eye_from_mouse,
                 near, far, width, height):
        self.ndc_from_eye = ndc_from_eye
        self.ndc_from_model = ndc_from_model
        self.eye_from_model = eye_from_model
        self.eye_from_mouse = eye_from_mouse
        self.near = near
        self.far = far
        self.width = width
        self.height = height

    @staticmethod
    def create(width, height, eye_from_model, bounding_box):
        left, right, bottom, top = GlTransformations.get_viewport_eye_rect(width, height)
        eye_from_model_matrix = eye_from_model.matrix()
        near, far, ndc_from_eye = GlTransformations.get_ndc_from_eye(
            left, right, bottom, top, eye_from_model, bounding_box)
        ndc_from_model = np.dot(ndc_from_eye, eye_from_model_matrix)
        eye_from_mouse = GlTransformations.get_eye_from_mouse(width, height)
        return GlTransformations(
            ndc_from_eye, ndc_from_model, eye_from_model_matrix,
            eye_from_mouse, near, far, width, height)

    @staticmethod
    def get_viewport_eye_rect(width, height):
        eye_from_mouse = GlTransformations.get_eye_from_mouse(width, height)
        tl = eye_from_mouse * pt.Point((0.0, 0.0, 0.0))
        br = eye_from_mouse * pt.Point((float(width), float(height), 0.0))
        return tl[0], br[0], tl[1], br[1]

    @staticmethod
    def get_eye_from_mouse(width, height):
        return pt.Transformation.translation((-width / 2.0, -height / 2.0, 0.0))

    @staticmethod
    def get_ndc_from_eye(left, right, bottom, top, eye_from_model, bounding_box):
        near, far = GlTransformations.get_near_far(eye_from_model, bounding_box)
        if abs(near - far) < 1e-9:
            # Not sure why this happened to someone, but avoid near == far
            near, far = -1e4, 1e4
        return near, far, GlTransformations.ortho_matrix(left, right, bottom, top, near, far)

    def get_ndc_from_eye_for_triad(self, near, far):
        return GlTransformations.ortho_matrix(
            -self.width / 2, self.width / 2, - self.height / 2, self.height / 2, near, far)

    @staticmethod
    def get_near_far(eye_from_model, bounding_box):
        near, far = -1e4, 1e4
        if bounding_box is not None and not bounding_box.is_empty():
            eye_bbox = eye_from_model * bounding_box
            near, far = eye_bbox.min[2], eye_bbox.max[2]
        return near, far

    @staticmethod
    def ortho_matrix(left, right, bottom, top, near, far):
        m = np.identity(4, dtype=np.float32)
        m[0, 0] = 2 / (right - left)
        m[1, 1] = 2 / (top - bottom)
        m[2, 2] = - 2 / (far - near)
        m[0, 3] = -(right + left) / (right - left)
        m[1, 3] = -(top + bottom) / (top - bottom)
        m[2, 3] = -(far + near) / (far - near)
        return np.ascontiguousarray(m, dtype=np.float32)

