from core.Core import Core
from PySide.QtGui import QApplication
from platform import system as pl_system
from thread import start_new_thread
from time import time
from os import path, getcwd
from sys import exit


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
        system = pl_system()
        if system == 'Darwin':
            return getcwd()
        elif system == 'Linux':
            return path.dirname(getcwd())
        else:
            return getcwd()

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
                start_new_thread(self.analyse, (query,))

                return True
            elif request == 'Quit':
                self.stop()

        except Exception as e:
            self.stop()
            print str(e)

    def analyse(self, query):
        try:
            start_time = time()
            sentence = self.form.components['labels_panel'].labels['sentence']
            result = self.form.components['labels_panel'].labels['result']
            elapsed_time = self.form.components['labels_panel'].labels['time']
            sentence.setText('The sentence is:' + query + '\n')
            result.setText('Analysing ...' + '\n')
            result.setText('Prediction: ' + self.core.predict(query))
            elapsed_time.setText('Elapsed Time: ' + str(round((time() - start_time) * 1000)) + ' ms')
            result.repaint()
            QApplication.processEvents()
        except Exception as e:
            self.stop()
            print e

    def set_form(self, form):
        self.form = form

    def stop(self):
        print 'Stop'
        exit(0)
