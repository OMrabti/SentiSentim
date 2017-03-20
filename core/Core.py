from graphlab import SArray, SFrame
from text_analysis import *
from core.helpers import get_absolute_path, get_config_param
import findspark
import ConfigParser

try:

    findspark.init()
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession, SQLContext
    from pyspark.sql.types import StructField, StringType, IntegerType, StructType

    print("Successfully imported Spark Modules")

except ImportError as e:
    print("Can not import Spark Modules", e)
    sys.exit(1)


class Core(object):
    # Data retrievial using tweepy
    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.absolute_path = get_absolute_path()
        self.config.read(self.absolute_path + '/config.ini')
        self.naive_base_model = get_config_param('Paths', 'naive_base_model')
        self.neural_net_model = get_config_param('Paths', 'neural_net_model')
        self.logistic_regression_model = get_config_param('Paths', 'logistic_regression_model')
        self.spark_context, self.sql_context = self.get_spark_setup()
        # Network properties
        self.max_iterations = self.config.get('Net', 'max_iterations')
        self.target = self.config.get('Net', 'target')
        self.metric = self.config.get('Net', 'metric').split(',')

        pass

    @staticmethod
    def convert_func(x):
        try:
            if float(x) > 0.5:
                return 2
            elif float(x) == 0.5:
                return 1
            else:
                return 0
        except Exception as e:
            print 'convert_func', e

    @staticmethod
    def get_spark_setup():
        # Setup
        """
        Sets up spark environement
        :return: spark_context, sql_context
        :rtype:
        """
        # conf = SparkConf().setAppName('Sentilyse').setMaster('spark://192.168.1.71:7077')
        spark_context = SparkContext()
        sql_context = SQLContext(spark_context)
        return spark_context, sql_context

    def rdd_to_dataframe(self, columns, rows, columns_types):
        """
        Converts an rdd to a DataFrame
        :param rdd:
        :type rdd: RDD
        :param name_column:
        :type name_column: str
        :param label_column:
        :type label_column: str
        :return: a DataFrame from the given rdd
        :rtype: DataFrame
        """
        fields = [StructField(field_name, columns_types[field_name], True) for field_name in columns]
        schema = StructType(fields)
        data = self.sql_context.createDataFrame(rows, schema)
        data.printSchema()
        return data

        # Generic function
        # Split the data into train and test

    def get_data(self):
        sentences = self.spark_context.textFile(
            self.absolute_path + "/resources/stanfordSentimentTreebank/datasetSentences.txt")
        sentiment_labels = self.spark_context.textFile(
            self.absolute_path + "/resources/stanfordSentimentTreebank/sentiment_labels.txt")

        id_sentence = sentences.map(lambda item: item.split('\t'))
        sentences_df = self.rdd_to_dataframe(['id', 'sentence'],
                                             id_sentence.map(lambda l: (int(l[0]), l[1])), {'id': StringType(),
                                                                                            'sentence': StringType()})
        # sentences_df.show()

        id_sentiment = sentiment_labels.map(lambda item: item.split('|'))
        labels_df = self.rdd_to_dataframe(['id', 'sentiment'], id_sentiment.map(lambda l: (int(l[0]) + 1, l[1])),
                                          {'id': StringType(),
                                           'sentiment': StringType()})

        # labels_df.show()

        result = sentences_df.join(labels_df, on="id", how='outer').select("id", sentences_df["sentence"],
                                                                           labels_df["sentiment"])

        result.show()
        return result

        # Generic function
        # returns spark_context, sql_context

    def data_frame_with_target(self, data_frame):
        """

        :param data_frame:
        :type data_frame: DataFrame
        :return:
        :rtype: SFrame
        """
        data_sframe = SFrame(data_frame.toPandas())
        sentiment_array = data_sframe.select_column('sentiment')
        target_array = []
        for x in sentiment_array:
            try:
                target_array.append(self.convert_func(x))
            except Exception as ex:
                print len(target_array), 'get_sentiments', x
                target_array.append(3)
                print ex

        data_sframe.add_column(SArray(target_array, dtype=int), name='target')
        print data_sframe
        return data_sframe.dropna()

    def get_sentiments(self, data_frame):
        """

        :param data_frame:
        :type data_frame: DataFrame
        :return:
        :rtype:
        """

        data_sframe = self.data_frame_with_target(data_frame)
        self.do_train_classify(data_sframe)

    def do_train_classify(self, data):
        """

        :param data:
        :type data: SFrame
        :return:
        :rtype:
        """
        print '>> do_train_classify', self.naive_base_model
        df = self.sql_context.createDataFrame(data.to_dataframe())
        train_naive_bayes_model(df.rdd, self.spark_context, self.naive_base_model)

        # classify_using_nn(
        #     self.rdd_to_dataframe(['label', 'name'], df.rdd.map(lambda line: (line['target'], line['sentence'])),
        #                           {'label': IntegerType(),
        #                            'name': StringType()}),
        #     self.spark_context,
        #     self.neural_net_model)

    def predict(self, sentence):
        x = predict_row(sentence, self.spark_context, self.naive_base_model)
        if x == 0:
            return 'Negative'
        elif x == 2:
            return 'Positive'
        else:
            return 'Neutral'
