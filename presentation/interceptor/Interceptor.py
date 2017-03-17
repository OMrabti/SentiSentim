from core.Core import Core
import thread
import time
from PySide.QtGui import QApplication
import sys
import os
import platform


class Interceptor(object):
    def __init__(self):
        self.form = None
        self.core = Core()
        self.absolute_path = self.get_absolute_path()
        pass

    @staticmethod
    def get_absolute_path():
        """
        This method gets the absolute path to the current working directory according to the operating system
        :return: The absolute path to the current working directory
        :rtype: basestring
        """
        system = platform.system()
        if system == 'Darwin':
            return os.getcwd()
        elif system == 'Linux':
            return os.path.dirname(os.getcwd())
        else:
            return os.getcwd()

    def intercept(self, request):
        """

        :param request: a string that determines the action to be executed
        :type request: basestring
        :return:
        :rtype:
        """
        print request
        try:
            if request == 'Analyse' or request == 'query':
                query = self.form.get_text_field('query').get_value()
                thread.start_new_thread(self.analyse, (query,))

                return True
            elif request == 'Stop':
                self.stop()

        except Exception as e:
            self.stop()
            print str(e)

    def analyse(self, query):
        try:
            start_time = time.time()
            sentence = self.form.components['labels_panel'].labels['sentence']
            result = self.form.components['labels_panel'].labels['result']
            elapsed_time = self.form.components['labels_panel'].labels['time']
            sentence.setText('The sentence is:' + query + '\n')
            result.setText('Analysing ...' + '\n')
            result.setText('Prediction: ' + self.core.predict(query))
            elapsed_time.setText('Elapsed Time: ' + str(round((time.time() - start_time) * 1000)) + ' ms')
            result.repaint()
            QApplication.processEvents()
        except Exception as e:
            self.stop()
            print e

    def set_form(self, form):
        self.form = form

    def stop(self):
        print 'Stop'
        sys.exit(0)
