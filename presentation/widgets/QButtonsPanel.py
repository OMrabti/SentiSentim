from PySide.QtGui import QWidget, QHBoxLayout, QPushButton
import PySide.QtCore as QtCore


class QButtonsPanel(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.layout = QHBoxLayout()
        self.interceptor = None
        self.buttons = {}

    def add_button(self, text):
        button = QPushButton(text, self)
        self.buttons.update({text: button})

        self.connect(button, QtCore.SIGNAL('clicked()'), lambda: self.interceptor.intercept(text))
        self.layout.addWidget(button)

    def add_buttons(self, buttons):
        for button in buttons:
            self.add_button(button)

    def set_layout(self, layout):
        self.layout = layout

    def set_interceptor(self, interceptor):
        self.interceptor = interceptor

    def build(self):
        self.setLayout(self.layout)
