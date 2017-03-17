from PySide.QtGui import QWidget, QVBoxLayout

from presentation.widgets.QField import QField as Field
from presentation.widgets.QButtonsPanel import QButtonsPanel as ButtonsPanel
from presentation.widgets.QLabelsPanel import QLabelsPanel as LabelsPanel


class QForm(QWidget):
    def __init__(self, title):
        QWidget.__init__(self)
        # Layout Init.
        self.setGeometry(300, 300, 600, 100)
        self.setWindowTitle(title)
        # Attributes
        self.components = {}
        self.container = QVBoxLayout()
        self.interceptor = None

    def add_text_field(self, label, interceptor):
        field = Field(label, interceptor)
        self.container.addWidget(field)
        field.build()
        self.components.update({label: field})

    def add_buttons_panel(self, texts, interceptor):
        buttons = ButtonsPanel()
        buttons.set_interceptor(interceptor)
        buttons.add_buttons(texts)
        buttons.build()
        self.container.addWidget(buttons)
        self.components.update({'buttons_panel': buttons})

    def add_labels_panel(self, texts):
        labels = LabelsPanel()
        labels.add_labels(texts)
        labels.build()
        self.container.addWidget(labels)
        self.components.update({'labels_panel': labels})

    def get_text_field(self, label):
        return self.components[label]

    def set_interceptor(self, interceptor):
        self.interceptor = interceptor

    def build(self):
        self.setLayout(self.container)
