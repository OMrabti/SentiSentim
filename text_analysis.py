import string
import os
import sys
import nltk

# nltk.download('punkt')
# nltk.download('porter')
# nltk.download("stopwords")
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
except ImportError as e:
    print("Can not import NLTK Modules", e)

try:
    import findspark

    findspark.init()

    # from pyspark import SparkConsentence
    from pyspark.mllib.feature import HashingTF
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel

    from pyspark.ml.classification import MultilayerPerceptronClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline
    from pyspark.ml import PipelineModel
    from pyspark.ml.feature import Tokenizer
    from pyspark.ml.feature import HashingTF as HTF

except ImportError as e:
    print("Can not import Spark Modules", e)
    sys.exit(1)

# Module-level global variables for the `tokenize` function below
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


# Function to break sentence into "tokens", lowercase them, remove punctuation and stopwords, and stem them
def tokenize(sentence):
    tokens = word_tokenize(sentence)
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    stemmed = [STEMMER.stem(w) for w in no_stopwords]
    return [w for w in stemmed if w]


def train_naive_bayes_model(data, spark_context, model_folder):
    """

    :param data:
    :type data: RDD
    :param spark_context: the current spark context
    :type spark_context: SparkContext
    :param model_folder:
    :type model_folder: basestring
    :return:
    :rtype:
    """
    data_hashed = prepare_data(data)
    # Split data 70/30 into training and test data sets
    train_hashed, test_hashed = data_hashed.randomSplit([0.9, 0.1])
    model = get_naive_bayes_model(spark_context, train_hashed, model_folder)
    print 'training Accuracy :'
    get_accuracy(model, train_hashed)
    print 'testing Accuracy :'
    get_accuracy(model, test_hashed)


def prepare_data(data):
    """

    :param data:
    :type data: RDD
    :return: data cleaned and hashed
    :rtype: RDD
    """
    # Extract relevant fields in dataset -- category target and sentence content
    data_pared = data.map(lambda line: (line['target'], line['sentence']))
    # Prepare sentence for analysis using our tokenize function to clean it up
    data_cleaned = data_pared.map(lambda (target, sentence): (target, tokenize(sentence)))
    # Hashing term frequency vectorizer with 50k features
    htf = HashingTF(50000)

    # Create an RDD of LabeledPoints using category labels as labels and tokenized, hashed sentence as feature vectors
    data_hashed = data_cleaned.map(lambda (target, sentence): LabeledPoint(target, htf.transform(sentence)))

    # Ask Spark to persist the RDD so it won't have to be re-created later
    # data_hashed.persist()
    return data_hashed


def get_naive_bayes_model(spark_context, train_hashed, model_folder):
    """

    :param spark_context: the current spark context
    :type spark_context: SparkContext
    :param train_hashed:
    :type train_hashed: DataFrame
    :param model_folder:
    :type model_folder: basestring
    :return: a trained Naive Bayes model
    :rtype: NaiveBayesModel
    """
    if not os.path.exists(model_folder):

        # Train a Naive Bayes model on the training data
        model = NaiveBayes.train(train_hashed)

        # Ask Spark to save the model so it won't have to be re-trained later
        model.save(spark_context, model_folder)
    else:
        model = NaiveBayesModel.load(spark_context, model_folder)
    return model


def get_accuracy(model, test_hashed):
    """

    :param model:
    :type model: NaiveBayesModel
    :param test_hashed:
    :type test_hashed: RDD
    :return: the accuracy of the given model
    :rtype: float
    """
    # Compare predicted labels to actual labels
    prediction_and_labels = test_hashed.map(lambda point: (model.predict(point.features), point.label))

    # Filter to only correct predictions
    correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)

    # Calculate and print accuracy rate
    accuracy = correct.count() / float(test_hashed.count())

    print "Classifier correctly predicted category " + str(accuracy * 100) + " percent of the time"
    return accuracy


def predict_row(sentence, spark_context, model_folder):
    """
    
    :param sentence: a sentence to be analysed 
    :type sentence: basestring
    :param spark_context: the current spark context 
    :type spark_context: SparkContext
    :param model_folder: 
    :type model_folder: basestring
    :return: 0.0 if the sentence is negative, 1 if the sentence is neutral and 2 if the sentence is positive
    :rtype: float
    """
    htf = HashingTF(50000)
    sentence_features = htf.transform(tokenize(sentence))
    model = NaiveBayesModel.load(spark_context, model_folder)
    prediction = model.predict(sentence_features)
    print 'prediction :', prediction
    return prediction


def get_data_transformers():
    """
    Creates Data Transformers
    :return: tokenizer, hasher, classifier
    :rtype: Tokenizer, HashingTF, MultilayerPerceptronClassifier
    """
    # Tokenizer : Splits each name into words
    tokenizer = Tokenizer(inputCol="name", outputCol="words")
    # HashingTF : builds term frequency feature vectors from text data
    hasher = HTF(inputCol=tokenizer.getOutputCol(), outputCol="features", numFeatures=8)
    """
        specify layers for the neural network:
        input layer of size 4 (features), two intermediate of size 5 and 4
        and output of size 3 (classes)
    """
    # Network params
    maxIter = 20
    layers = 8, 5, 4, 5, 2
    blockSize = 128
    seed = 1234
    # Creating the trainer and set its parameters
    classifier = MultilayerPerceptronClassifier(maxIter=maxIter,
                                                layers=layers,
                                                blockSize=blockSize,
                                                seed=seed)
    return tokenizer, hasher, classifier


def classify_using_nn(data, spark_context, model_folder):
    # TODO: review how do we transform data to RDD
    # Splitting data
    train, test = data.randomSplit([0.6, 0.4], 1234)
    # if the model is already trained and saved
    # we just load it
    if not os.path.exists(model_folder):
        # Transforming data
        tokenizer, hasher, classifier = get_data_transformers()

        # Estimating data
        pipeline = Pipeline(stages=[tokenizer, hasher, classifier])

        # Train the model
        model = pipeline.fit(train)
        model.save(model_folder)
    else:
        model = PipelineModel.load(model_folder)

    # Compute accuracy on the train set
    print("Train set accuracy = {:.3g}.".format(evaluate(model, train)))

    # Compute accuracy on the test set
    print("Test set accuracy = {:.3g}.".format(evaluate(model, test)))

    spark_context.stop()


def evaluate(model, data):
    """
    Evaluates the model
    :param model: the trained model
    :type model: PipelineModel
    :param data:
    :type data: DataFrame
    :return: the accuracy of the model
    :rtype: float
    """
    evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                                  predictionCol="prediction",
                                                  metricName="accuracy")

    # Compute accuracy on the train set
    result = model.transform(data)
    result.show()
    return evaluator.evaluate(result)
