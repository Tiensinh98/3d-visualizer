import os
import sys
import point as pt


class View:
    def __init__(self, x, y, width, height, eye_from_model):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.eye_from_model = eye_from_model

    @staticmethod
    def create_default():
        # This seems to be only used once, in gl_manager, and never again.
        w, h = 640, 480
        eye_from_model = pt.Transformation(
            rotation=(1.0, 0.0, 0.0, 0.0),
            translation=(0, 0, 0),
            scale=1.0)
        return View(0, 0, w, h, eye_from_model)

    def get_eye_from_mouse(self):
        return pt.Transformation.translation((-self.width / 2.0, -self.height / 2.0, 0.0))

    def model_from_mouse(self):
        return self.eye_from_model.inv() * self.get_eye_from_mouse()

    def mouse_from_model(self):
        return self.get_eye_from_mouse().inv() * self.eye_from_model

    def resize(self, x, y, width, height):
        return View(x, y, width, height, self.eye_from_model)