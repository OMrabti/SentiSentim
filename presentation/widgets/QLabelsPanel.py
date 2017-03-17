from PySide.QtGui import QWidget, QVBoxLayout, QLabel
import PySide.QtCore as QtCore


class QLabelsPanel(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.layout = QVBoxLayout()
        self.labels = {}

    def add_label(self, text):
        label = QLabel(text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(label)
        self.labels.update({text: label})
        return label

    def add_labels(self, labels):
        for label in labels:
            self.add_label(label)

    def set_layout(self, layout):
        self.layout = layout

    def build(self):
        self.setLayout(self.layout)
