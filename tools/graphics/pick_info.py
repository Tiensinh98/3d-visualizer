import numpy as np
from . import gl

__all__ = ['PickInfo']


class PickInfo:
    __slots__ = ['width', 'height', 'pick_gl_id', 'depth_gl_id', 'frame_buffer_id']

    def __init__(self, width, height, pick_gl_id, depth_gl_id, frame_buffer_id):
        self.width = width
        self.height = height
        self.pick_gl_id = pick_gl_id
        self.depth_gl_id = depth_gl_id
        self.frame_buffer_id = frame_buffer_id

    @staticmethod
    def create(width, height):
        pick_gl_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, pick_gl_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        depth_gl_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, depth_gl_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, width,
                        height, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        frame_buffer_id = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, frame_buffer_id)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, pick_gl_id, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                  gl.GL_TEXTURE_2D, depth_gl_id, 0)
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        assert status == gl.GL_FRAMEBUFFER_COMPLETE, 'Invalid status: {}'.format(status)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return PickInfo(width, height, pick_gl_id, depth_gl_id, frame_buffer_id)

    def pick(self, x, y, pick_to_color):
        x, y = int(round(x)), int(round(y))
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frame_buffer_id)
        pixels = gl.glReadPixels(x, y, 1, 1, gl.GL_RGB, gl.GL_FLOAT)
        rgb = [int(v) for v in np.round(255.0 * pixels[0, 0])]
        idx = decode_padic(rgb)
        obj = None
        if idx != 0:
            obj = pick_to_color.get(idx, None)
            if obj is None:
                print('unknown pick', rgb)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        print('RBG:', obj)
        return obj

    def resize(self, width, height):
        gl.glDeleteFramebuffers(1, [self.frame_buffer_id])
        gl.glDeleteTextures([self.pick_gl_id])
        gl.glDeleteTextures([self.depth_gl_id])
        pick_info = PickInfo.create(width, height)
        self.pick_gl_id = pick_info.pick_gl_id
        self.depth_gl_id = pick_info.depth_gl_id
        self.frame_buffer_id = pick_info.frame_buffer_id
        self.width = width
        self.height = height


def decode_padic(rgb, b=256, idx=0):
    m = 1
    for i in range(3):
        idx += m * rgb[i]
        m *= b
    return idx
