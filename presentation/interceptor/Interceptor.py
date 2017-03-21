from core.Core import Core
from PySide.QtGui import QApplication
from platform import system as pl_system
from thread import start_new_thread
from time import time
from os import path, getcwd
from sys import exit
from langdetect import detect
from microsofttranslator import Translator
from core.helpers import json_file_to_object, get_absolute_path, chunkstring

translator = Translator('sentilyse', '8DqdkAzW8UN9ClRTuBoy0lS5DuFkkh/vMVgnaLcXKQY=')
language_codes = json_file_to_object(get_absolute_path() + '/resources/language-codes.json')


class Interceptor(object):
    def __init__(self):
        self.form = None
        self.core = Core()
        self.absolute_path = self.get_absolute_path()
        # self.core.train_nb_classifier() last testing accuracy = 62 percent
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
            if request == 'Analyse' or request == 'Query':
                query = self.form.get_text_field('Query').get_value().lower()
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
            language = self.form.components['labels_panel'].labels['language']
            result = self.form.components['labels_panel'].labels['result']
            elapsed_time = self.form.components['labels_panel'].labels['time']
            sentence.setText('Sentence in english:  Loading ...')
            language.setText('Original language:  Loading ...')
            result.setText('Prediction: Loading ...')
            elapsed_time.setText('Elapsed Time:  Loading ...')
            for lang in language_codes:
                if lang['alpha2'] == detect(query):
                    language.setText('Original language: ' + lang['English'] + '\n')

            translated_query = translator.translate(query, 'en', detect(query))
            chunked_text = ''
            for chunk in chunkstring('Sentence in english: ' + translated_query, 90):
                chunked_text += (chunk + '\n')
            sentence.setText(chunked_text + '\n')
            sentence.repaint()
            result.repaint()
            result.setText('Prediction: ' + self.core.predict(translated_query))
            elapsed_time.setText('Elapsed Time: ' + str(round((time() - start_time) * 1000)) + ' ms')
            result.repaint()
            QApplication.processEvents()
        except Exception as e:
            print e
            self.stop()

    def set_form(self, form):
        self.form = form

    def stop(self):
        print 'Stop'
        exit(0)
