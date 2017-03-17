from PySide.QtGui import QWidget, QHBoxLayout, QLabel, QLineEdit
import PySide.QtCore  as QtCore


class QField(QWidget):
    def __init__(self, label_text, interceptor):
        QWidget.__init__(self)
        self.container = QHBoxLayout()
        self.label = QLabel(label_text + ' :')
        self.text_field = QLineEdit()
        self.interceptor = interceptor
        self.connect(self.text_field, QtCore.SIGNAL('returnPressed()'), lambda: self.interceptor.intercept(label_text))
        self.container.addWidget(self.label)
        self.container.addWidget(self.text_field)

    def set_value(self, value):
        self.text_field.setText(value)

    def get_value(self):
        return self.text_field.text()

    def get_text_component(self):
        return self.text_field

    def build(self):
        self.setLayout(self.container)
