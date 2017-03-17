import tweepy
import ConfigParser
from pymongo import MongoClient
import subprocess
import platform
import os

mongod = ''  # get_mongod()


def get_mongod():
    return subprocess.Popen(['mongod'])


def get_absolute_path():
    """
    Gets the absolute path of the working directory
    :return: the absolute path of the working directory
    :rtype: str
    """
    system = platform.system()
    if system == 'Darwin':
        return os.getcwd()
    elif system == 'Linux':
        return os.path.dirname(os.getcwd())
    else:
        return os.getcwd()


def get_config():
    config = ConfigParser.ConfigParser()
    config.read(get_absolute_path() + '/config.ini')
    return config


# Generic function
def get_config_param(section, option):
    """
    Gets parameters form the config file (uses get_absolute_path())
    :param section: section
    :type section: str
    :param option: option
    :type option: str
    :return: a parameter given the section and the option
    :rtype: str
    """
    param = get_config().get(section, option)
    if (param.find('/') > -1) or (section.lower().find('path') > -1):
        return get_absolute_path() + param
    else:
        return param


def get_db():
    print 'mongod pid : ', mongod.pid
    client = MongoClient()
    db = client.tweety
    return db


def get_tweepy_api():
    consumer_token = get_config_param('Tweepy', 'consumer_token')
    consumer_secret = get_config_param('Tweepy', 'consumer_secret')
    access_token = get_config_param('Tweepy', 'access_token')
    access_token_secret = get_config_param('Tweepy', 'access_token_secret')

    auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    # api.update_status('tweepy + oauth!')
    return api


def get_tweets(query):
    db = get_db()
    print 'get_tweets', query
    try:
        # print len(api.search(q=query + 'and security'))
        api = get_tweepy_api()
        for result in tweepy.Cursor(api.search,
                                    q=query,
                                    count=100,
                                    result_type="recent",
                                    include_entities=True,
                                    lang="en").items():
            print result.text
            if query.lower() in result.text.lower():
                db[query.lower().lstrip()].insert_one(result._json)
    except Exception as e:
        print e
        pass


def find_all_tweets(query):
    db = get_db()
    text_list = []
    try:
        for post in db[query].find():
            text_list.append(post['text'])
        return text_list
    except Exception as e:
        mongod.kill()
        print str(e.message)
