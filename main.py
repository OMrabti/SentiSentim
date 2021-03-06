import sys
from presentation.QForm import QForm
from PySide.QtGui import QApplication
from presentation.interceptor.Interceptor import Interceptor as Interceptor


class Main:
    def __init__(self):
        app = QApplication(sys.argv)
        print 'Main----'
        form = QForm('Sentilyse')
        self.interceptor = Interceptor()
        self.interceptor.set_form(form)

        form.add_text_field('Query', self.interceptor)
        form.add_labels_panel(['sentence', 'language', 'result', 'time'])
        form.add_buttons_panel(['Quit', 'Analyse'], self.interceptor)

        form.build()
        form.show()
        sys.exit(app.exec_())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.interceptor.stop()


if __name__ == '__main__':
    main = Main()
