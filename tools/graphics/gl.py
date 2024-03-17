# Copyright (C) 2015 Akselos
try:
    import OpenGL.GLU  # bug in pyopengl < 3.0.2: https://bugs.python.org/issue26245
    from OpenGL.GL.framebufferobjects import *
    from OpenGL.GL import *
    from OpenGL.arrays import ArrayDatatype as ad
    from OpenGL.GL import shaders
    import OpenGL.arrays.numpymodule as nm
except ImportError:
    # Sometimes the server uses GUI code, but may not have OpenGL installed.
    # Just ignore the import error.  We do something similar for qt.py.
    print('\nDid not import OpenGL module')
