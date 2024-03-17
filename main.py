import sys
import os
import qt
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), 'tools'))
import graphics as gi
import model as md
import ui_state as ust


class MainWindow(qt.QMainWindow):
    open_button: qt.QPushButton
    view_fit_button: qt.QPushButton
    save_image_button: qt.QPushButton
    field_name_cbb: qt.QComboBox
    selection_mode_cbb: qt.QComboBox
    discrete_check_box = qt.QCheckBox

    def __init__(self, model_state_jdata, application_settings, ui_state):
        super().__init__()
        self.model_state_jdata = model_state_jdata
        self.application_settings = application_settings
        self.ui_state = ui_state
        self.canvas = gi.Canvas(self)
        window_title = application_settings.get('window_title')
        self.init_gui(window_title)

    @staticmethod
    def init_window(model_state_jdata, application_settings):
        model_state = md.ModelState.create(model_state_jdata)
        ui_state = ust.UiState.create(model_state, application_settings)
        component_system = model_state.component_system
        ui_state.set_window_entity(component_system)
        return MainWindow(model_state_jdata, application_settings, ui_state)

    def change_field_name(self):
        self.canvas.is_render_info_dirty = True
        field_name = self.field_name_cbb.currentData(qt.Qt.UserRole)
        self.set_application_settings('field_name', field_name)
        self.canvas.update()

    def set_discrete(self, state):
        self.set_application_settings('is_discrete_color', state)
        self.canvas.update()

    def save_pixels_to_image(self, filepath):
        pixels = self.canvas.grab_pixels()
        height, width = self.canvas.height, self.canvas.width
        pixels = np.array(np.round(255 * pixels), dtype=np.uint8).reshape((height, width, 3))[::-1, :, :]
        background_color = (255, 255, 255)
        image = Image.fromarray(pixels, 'RGB')
        image.save(filepath, transparency=background_color)

    def change_selection_mode(self):
        self.canvas.is_render_info_dirty = True
        selection_mode = self.selection_mode_cbb.currentData(qt.Qt.UserRole)
        self.set_application_settings('selection_mode', selection_mode)
        self.canvas.update()

    def init_gui(self, window_title):
        self.open_button = qt.QPushButton('Open')
        self.field_name_cbb = qt.QComboBox()
        selection_mode_label = qt.QLabel('Selection Mode')
        self.selection_mode_cbb = qt.QComboBox()
        self.selection_mode_cbb.addItem('Components', 'component')
        self.selection_mode_cbb.addItem('Solutions', 'solution')
        self.selection_mode_cbb.activated.connect(self.change_selection_mode)
        self.view_fit_button = qt.QPushButton('Fit')
        self.save_image_button = qt.QPushButton('Save Image')
        self.discrete_check_box = qt.QCheckBox('Discrete color')
        self.view_fit_button.clicked.connect(self.view_fit)
        self.save_image_button.clicked.connect(self.save_image)
        self.field_name_cbb.activated.connect(self.change_field_name)
        self.open_button.clicked.connect(self.insert_component)
        self.discrete_check_box.stateChanged[int].connect(self.set_discrete)
        self.setWindowTitle(window_title)
        central_widget = qt.QWidget()
        gui_layout = qt.QVBoxLayout()
        tool_layout = qt.QGridLayout()
        tool_layout.addWidget(self.open_button, 0, 0)
        tool_layout.addWidget(self.field_name_cbb, 0, 1)
        tool_layout.addWidget(self.save_image_button, 1, 0)
        tool_layout.addWidget(self.view_fit_button, 1, 1)
        tool_layout.addWidget(selection_mode_label, 2, 0, qt.Qt.AlignCenter)
        tool_layout.addWidget(self.selection_mode_cbb, 2, 1)
        tool_layout.addWidget(self.discrete_check_box, 3, 0)
        gui_layout.addLayout(tool_layout)
        central_widget.setLayout(gui_layout)
        self.setCentralWidget(central_widget)
        gui_layout.addWidget(self.canvas)
        self.set_application_settings('selection_mode', 'components')
        self.set_application_settings('field_name', None)
        self.set_application_settings('field_names', [])
        self.set_application_settings('is_discrete_color', 0)

    def set_application_settings(self, key, value):
        self.application_settings[key] = value

    @qt.Slot()
    def insert_component(self):
        option = qt.QFileDialog.Options()
        filepath = qt.QFileDialog.getOpenFileName(self, "Open File", "/home", "CSV Files (*.exo)", options=option)
        if os.path.isfile(filepath[0]):
            component_jdata = md.Component.create_json_for_insert(filepath[0])
            model_state = self.ui_state.model_state
            component_system = model_state.component_system
            component = md.Component.create(component_jdata, component_system)
            model_state.insert_component(component)
            field_names = component.get_field_names()
            existing_field_names = self.application_settings['field_names']
            if len(field_names):
                self.field_name_cbb.clear()
                existing_field_names = list(set(existing_field_names + field_names))
                for field_name in existing_field_names:
                    gui_name = ''.join([name.capitalize() for name in field_name.split('_')])
                    self.field_name_cbb.addItem(gui_name, field_name)
            self.ui_state.set_window_entity(component_system)
            self.set_application_settings('selection_mode', 'components')
            current_field_name = self.field_name_cbb.currentData(qt.Qt.UserRole)
            self.set_application_settings('field_name', current_field_name)
            self.set_application_settings('field_names', existing_field_names)
            self.canvas.is_render_info_dirty = True
            self.view_fit()

    @qt.Slot()
    def save_image(self):
        option = qt.QFileDialog.Options()
        filepath = qt.QFileDialog.getSaveFileName(self, "Save File", "image", "PNG Files (*.png)", options=option)
        filepath = filepath[0]
        self.save_pixels_to_image(filepath)
        print("File saved", filepath[0])

    def view_fit(self):
        self.canvas.view_fit()

    def closeEvent(self, *args, **kwargs):
        self.canvas.delete_buffers()


if __name__ == '__main__':
    app = qt.QApplication([sys.argv[0], '-style', 'fusion'])
    app_settings_json = {
        'window_title': 'OpenGL Application',
        'selection_mode': 'components'
    }
    model_json = {}
    window = MainWindow.init_window(model_json, app_settings_json)
    window.show()
    app.exec_()
