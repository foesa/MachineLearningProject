import pymongo
import preprocessor as prepro
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import json


def mongo_downloader():
    CONNECTION_STRING = "mongodb+srv://foesa:Random12@cluster0.sdvcb.mongodb.net/TweetDB?retryWrites=true&w=majority"
    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.get_database('TweetDB')
    records = db.TweetsData
    tweets = records.find({})
    total_tweets = []
    for i in tweets:
        total_tweets.append(i)
    return total_tweets


def is_one_canditate_mentioned(tweet):
    # TODO: Check that candiate is mentioned
    trumps_names = ["donald", "trump"]
    bidens_names = ["joe", "biden"]
    opponent_names = {
        "Donald Trump lang:en": bidens_names,
        "Trump lang:en": bidens_names,
        "Joe Biden lang:en": trumps_names,
        "Biden lang:en": trumps_names,
    }
    candidate_names = {
        "Donald Trump lang:en": trumps_names,
        "Trump lang:en": trumps_names,
        "Joe Biden lang:en": bidens_names,
        "Biden lang:en": bidens_names,
    }
    contents = tweet["text"].lower()
    prepro.set_options(prepro.OPT.URL, prepro.OPT.HASHTAG)
    clean_contents = prepro.clean(contents)

    for opponent_name in opponent_names[tweet["Candidate"]]:
        if clean_contents.find(opponent_name) != -1:
            return False

    for name in candidate_names[tweet["Candidate"]]:
        if clean_contents.find(name) != -1:
            return True

    return False


def clean_tweet(tweet):
    contents = tweet["text"].lower()
    # May want to change these
    prepro.set_options(
        prepro.OPT.URL,
        prepro.OPT.EMOJI,
        prepro.OPT.SMILEY,
        prepro.OPT.NUMBER
    )
    clean_contents = prepro.clean(contents)
    tweet["text"] = clean_contents


def create_tokens(tweet):
    tokens = word_tokenize(tweet["text"])
    stop_words = set(stopwords.words("english"))
    useful_tokens = []
    lemma = WordNetLemmatizer()
    for token in tokens:
        if (not token in stop_words) and (token.isalpha()):
            lemmatnised_token = lemma.lemmatize(token)
            useful_tokens.append(lemmatnised_token)
    tweet["tokens"] = useful_tokens


def get_processed_tweets():
    raw_tweets = mongo_downloader()
    processed_tweets = []
    for tweet in raw_tweets:
        if is_one_canditate_mentioned(tweet):
            clean_tweet(tweet)
            create_tokens(tweet)
            if tweet["Candidate"] == ('Biden lang:en' or 'Joe Biden lang:en'):
                tweet["Candidate"] = 'Joe Biden'
            else:
                tweet["Candidate"] = 'Donald Trump'
            if tweet["sentiment"] != "Remove":
                processed_tweets.append({
                    "candidate": tweet["Candidate"],
                    "tokens": tweet["tokens"],
                    "sentiment": tweet["sentiment"],
                    # Add any other useful tweet data
                })
    return processed_tweets


def main():
    # The line below must be run the first time you run the program
    # nltk.download()
    tweets = get_processed_tweets()
    with open("tweet_tokens.txt", "w") as f:
        f.write(json.dumps(tweets))


if __name__ == '__main__':
    main()
