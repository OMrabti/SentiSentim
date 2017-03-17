from presentation.QForm import QForm
import sys
from PySide.QtGui import *
from presentation.interceptor.Interceptor import Interceptor as Interceptor


class Main:
    def __init__(self):
        app = QApplication(sys.argv)
        print 'Main----'
        form = QForm('TweeLab')
        self.interceptor = Interceptor()
        self.interceptor.set_form(form)

        form.add_text_field('query', self.interceptor)
        form.add_labels_panel(['sentence', 'result', 'time'])
        form.add_buttons_panel(['Stop', 'Analyse'], self.interceptor)

        form.build()
        form.show()
        sys.exit(app.exec_())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.interceptor.stop()


if __name__ == '__main__':
    main = Main()
