import weakref
import hashlib
from ctypes import c_void_p
import numpy as np

from . import shaders as sd
from . import gl

NULL = c_void_p(0)

__all__ = ['BodyBuffer']

BUFFERS_TO_DELETE = []
VAOS_TO_DELETE = []


def free_buffers():
    if len(VAOS_TO_DELETE) > 0:
        gl.glDeleteVertexArrays(len(VAOS_TO_DELETE), VAOS_TO_DELETE)
    VAOS_TO_DELETE[:] = []
    if len(BUFFERS_TO_DELETE) > 0:
        gl.glDeleteBuffers(len(BUFFERS_TO_DELETE), BUFFERS_TO_DELETE)
    BUFFERS_TO_DELETE[:] = []


class GlIdHolder:

    GL_ID_CACHE = weakref.WeakValueDictionary()

    def __init__(self, gl_id, n_bytes):
        self.gl_id = gl_id
        self.n_bytes = n_bytes

    @staticmethod
    def create(array, gl_type):
        assert isinstance(array, np.ndarray)
        hash_key = hashlib.sha512(array.data).hexdigest()
        array_n_bytes = gl.ad.arrayByteCount(array)
        key = gl_type, array_n_bytes, hash_key
        gl_buffer_id_cache = GlIdHolder.GL_ID_CACHE.get(key, None)
        if gl_buffer_id_cache is not None:
            return gl_buffer_id_cache
        data_pointer = gl.ad.voidDataPointer(array)
        buffer_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl_type, buffer_id)
        gl.glBufferData(gl_type, array_n_bytes, data_pointer, gl.GL_STATIC_DRAW)
        gl_error = gl.glGetError()
        if gl_error == gl.GL_OUT_OF_MEMORY:
            raise MemoryError()
        elif gl_error != gl.GL_NO_ERROR:
            raise ValueError()
        gl_buffer_id = GlIdHolder(buffer_id, array_n_bytes)
        GlIdHolder.GL_ID_CACHE[key] = gl_buffer_id

        return gl_buffer_id

    def __del__(self):
        BUFFERS_TO_DELETE.append(self.gl_id)


class Buffer:
    __slots__ = ['gl_id_holder', 'length', 'gl_type', 'gl_data_type', 'attrib_info']

    def __init__(self, gl_id_holder, length, gl_type, gl_data_type, attrib_info):
        self.gl_id_holder = gl_id_holder
        self.length = length
        self.gl_type = gl_type
        self.gl_data_type = gl_data_type
        self.attrib_info = attrib_info

    @staticmethod
    def create(array, gl_type, attrib_info=None):
        assert isinstance(array, np.ndarray)
        gl_data_type = gl.nm.ARRAY_TO_GL_TYPE_MAPPING[array.dtype.type(0).dtype]
        gl_id_holder = GlIdHolder.create(array, gl_type)
        gl.glBindBuffer(gl_type, 0)
        return Buffer(gl_id_holder, len(array), gl_type, gl_data_type, attrib_info)

    def get_gl_id_holder(self):
        return self.gl_id_holder


class Vao:
    __slots__ = ['gl_array_id', 'vertex_buffer', 'attrib_buffers', 'indices_buffer']

    def __init__(self, gl_array_id, vertex_buffer, attrib_buffers, indices_buffer):
        self.gl_array_id = gl_array_id
        self.vertex_buffer = vertex_buffer
        self.attrib_buffers = attrib_buffers
        self.indices_buffer = indices_buffer

    @staticmethod
    def create(vertex_buffer, attrib_buffers, indices_buffer):
        num_elem = 3
        gl_array_id = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(gl_array_id)

        gl_vertices_dtype = vertex_buffer.gl_data_type
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer.gl_id_holder.gl_id)
        gl.glEnableVertexAttribArray(sd.VERTEX_LOCATION)
        gl.glVertexAttribPointer(
            sd.VERTEX_LOCATION, num_elem, gl_vertices_dtype, gl.GL_FALSE, 0, NULL)

        for attrib_name, attrib_buffer in attrib_buffers.items():
            gl_attrib_dtype = attrib_buffer.gl_data_type
            location_idx = attrib_buffer.attrib_info.location_idx
            n_attrib_components = attrib_buffer.attrib_info.n_components
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, attrib_buffer.gl_id_holder.gl_id)
            gl.glEnableVertexAttribArray(location_idx)
            gl.glVertexAttribPointer(
                location_idx, n_attrib_components, gl_attrib_dtype, gl.GL_FALSE, 0, NULL)

        if indices_buffer is not None:
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, indices_buffer.gl_id_holder.gl_id)

        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        return Vao(gl_array_id, vertex_buffer, attrib_buffers, indices_buffer)

    def __del__(self):
        VAOS_TO_DELETE.append(self.gl_array_id)


class BodyBuffer:
    __slots__ = ['vertex_buffer', 'attrib_buffers', 'indices_buffer', 'vao', 'primitive_mode', 'bounding_box']

    def __init__(self, vertex_buffer, attrib_buffers, indices_buffer, vao, primitive_mode):
        self.vertex_buffer = vertex_buffer
        self.attrib_buffers = attrib_buffers
        self.indices_buffer = indices_buffer
        self.vao = vao
        self.primitive_mode = primitive_mode

    @staticmethod
    def create(vertices, attrib_values, indices):
        assert len(indices.shape) == 2
        gl_type = gl.GL_ARRAY_BUFFER
        primitive_mode = indices.shape[1]
        indices = np.ascontiguousarray(indices.flatten(), dtype=np.uint32)
        assert len(vertices.shape) == 2
        assert vertices.shape[-1] == 3
        assert vertices.dtype == np.dtype('float32')
        vertex_buffer = Buffer.create(vertices, gl_type)
        attrib_buffers = {}
        for attrib_name, attrib_array in attrib_values.items():
            attrib_info = sd.ATTRIB_INFOs[attrib_name]
            assert len(attrib_array) == len(vertices)
            assert len(attrib_array.shape) == 2 or attrib_info.n_components == 1
            if len(attrib_array.shape) == 2:
                assert attrib_array.shape[-1] == attrib_info.n_components
            attrib_buffers[attrib_name] = Buffer.create(attrib_array, gl_type, attrib_info)

        assert np.min(indices) >= 0, indices
        assert np.max(indices) < len(vertices)
        indices_buffer = Buffer.create(indices, gl.GL_ELEMENT_ARRAY_BUFFER)
        vao = Vao.create(vertex_buffer, attrib_buffers, indices_buffer)
        return BodyBuffer(vertex_buffer, attrib_buffers, indices_buffer, vao, primitive_mode)

    def bind(self):
        gl.glBindVertexArray(self.vao.gl_array_id)

    def unbind(self):
        gl.glBindVertexArray(0)
