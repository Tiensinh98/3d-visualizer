try:
    from PySide2.QtWidgets import *
    from PySide2.QtCore import *
    from PySide2.QtQml import *
    from PySide2.QtGui import *
    from PySide2.QtQuick import *
    from PySide2.QtOpenGL import *
except ImportError as e:
    # Sometimes the server uses GUI code, but may not have qt installed.
    # We use qt in a few random places in the GUI code (e.g. message boxes,
    # enum values), so qt may get imported even when we don't need it.
    # I guess we could remove these random places, but instead just ignore
    # import error.
    print('\nDid not import qt module:\n%s' % e)

def is_checked(box):
    return box.checkState() == Qt.Checked
