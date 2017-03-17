from PySide.QtGui import QWidget, QVBoxLayout, QLabel, QMessageBox, QPalette, QSizePolicy

from presentation.widgets.QLabelsPanel import QLabelsPanel as LabelsPanel

from presentation.widgets.QButtonsPanel import QButtonsPanel as ButtonsPanel


class QDataViewer(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        # Layout Init.
        self.setGeometry(300, 300, 600, 100)
        self.setWindowTitle('ArganRecogn')
        # Attributes
        self.filename = ""
        self.upload_destination = ""

        self.layout = QVBoxLayout()
        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored,
                                      QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        self.labels_panel = LabelsPanel()
        self.buttons_panel = ButtonsPanel()

    def add_button(self, text, action):
        self.buttons_panel.add_button(text, action)

    def add_label(self, text):
        return self.labels_panel.add_label(text)

    @staticmethod
    def show_alert_dialog(exception):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)

        msg.setText("Something wrong occurred")
        msg.setInformativeText(str(exception.message))
        msg.setWindowTitle("Something wrong occurred")
        msg.setDetailedText("The details are as follows:\n" + exception.message)
        print "The details are as follows:\n" + exception.message
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.buttonClicked.connect(QDataViewer.msgbtn)

        retval = msg.exec_()
        print "value of pressed message box button:", retval

    @staticmethod
    def show_dialog(message, title):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText(title)
        msg.setInformativeText(message)
        msg.setWindowTitle("Something wrong occurred")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.buttonClicked.connect(QDataViewer.msgbtn)

        retval = msg.exec_()
        print "value of pressed message box button:", retval

    @staticmethod
    def msgbtn(i):
        print "Button pressed is:", i.text()

    def build(self):
        self.labels_panel.build()
        self.buttons_panel.build()
        self.layout.addWidget(self.labels_panel)
        self.layout.addWidget(self.buttons_panel)
        self.setLayout(self.layout)
