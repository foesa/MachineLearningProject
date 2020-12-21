# -*- coding: utf-8 -*-
"""
Author John Malone-Leigh

Date Created: 14/12/2020


1) This script downloads the sentiment classified twitter data
2) Preprocesses the Twitter data
3) Uses a Naive Bayes Classifier to train a model to predict sentiment

Modules:
    o mongo_downloader - Downloads tweets to analyse
    o twitter_samp - Downloads general tweets to use as baseline
    o lemmatize_sentence - 
    o remove_noise - Removes noise like links
    o get_tweets_for_model - changes strings to dictionaries
    o pre_process1 - Preprocesses the tweets to analyse. 
        -Does this by removing tweets which mention both Candiates
        -Can change settings to include Both, or one candidate
    o pre_process2 - Preprocessing to allow a model to be trained
    o cross_val - Using K-fold cross validation to evaluate data
    o evaluation - Returns f scores, precision and recall
    o naive_bayes_model - Brings all modules together, trains model

"""

import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
from nltk.corpus import stopwords
import random
from nltk import classify
from nltk import NaiveBayesClassifier
import pymongo
import collections
import numpy as np

def mongo_downloader():

    CONNECTION_STRING = "mongodb+srv://foesa:Random12@cluster0.sdvcb.mongodb.net/TweetDB?retryWrites=true&w=majority"
    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.get_database('TweetDB')
    records = db.TweetsData
    tweets = records.find({})
    #length = tweets.count() 
    total_tweets=[]
    for i in tweets:
        #print(i)
        total_tweets.append(i)
    
    maxn=len(total_tweets)
    folder="project_datafinal.txt" #Set to where you want to save file
    f=open(folder,'w',encoding='utf8')
    for i in range(0,maxn,1):
        x=str(total_tweets[i])+'\n'
        f.write(x)
    f.close()

def twitter_samp():
    nltk.download('twitter_samples')
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    return positive_tweets, negative_tweets


def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def pre_process2(positive_tweet_tokens,negative_tweet_tokens,
                 neutral_tweet_tokens):
        
    stop_words = stopwords.words('english')

    
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    neutral_cleaned_tokens_list = []
    
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
        
    for tokens in neutral_tweet_tokens:
        neutral_cleaned_tokens_list.append(remove_noise(tokens, stop_words))          

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    neutral_tokens_for_model = get_tweets_for_model(neutral_cleaned_tokens_list)    
    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]
    
    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]
    neutral_dataset = [(tweet_dict, "Neutral")
                         for tweet_dict in neutral_tokens_for_model]    

    dataset = positive_dataset + negative_dataset+neutral_dataset
    
    random.shuffle(dataset)
    train_data = dataset[int(len(dataset)/5):]
    test_data = dataset[:int(len(dataset)/5)]

    return train_data, test_data


def pre_process_1(candidates):
    folder='project_datafinal.txt'
    f=open(folder,'r',encoding='utf8')
    data=f.readlines()
    f.close()
        
    text='text'
    date='date' 
    total_positive=[]
    total_negative=[]
    total_neutral=[]
    for i in data:
        start=i.find(text)
        end=i.find(date)
        #Making sure tweet doesn'tinclude Both candiates
        if "Biden" in candidates:
            if "Biden lang:en" in i and "Trump" not in i:
                    if "'sentiment': 'Negative'" in i:
                        total_negative.append(i[start+7:end-3])
                    if "'sentiment': 'Positive'" in i:
                        total_positive.append(i[start+7:end-3])
                    if "'sentiment': 'Neutral'" in i:
                        total_neutral.append(i[start+7:end-3]) 
        if "Trump" in candidates:
            if "Trump lang:en" in i and "Biden" not in i:
                    if "'sentiment': 'Negative'" in i:
                        total_negative.append(i[start+7:end-3])
                    if "'sentiment': 'Positive'" in i:
                        total_positive.append(i[start+7:end-3])
                    if "'sentiment': 'Neutral'" in i:
                        total_neutral.append(i[start+7:end-3]) 
                    
    positive_tokens=[]
    negative_tokens=[]
    neutral_tokens=[]
    #Tokenizing tweets
    for i in total_positive:   
        positive_token=nltk.word_tokenize(i)
        positive_tokens.append(positive_token)
    for i in total_negative:
        negative_token=nltk.word_tokenize(i)
        negative_tokens.append(negative_token)
    for i in total_neutral:
        neutral_token=nltk.word_tokenize(i)
        neutral_tokens.append(neutral_token)

    return positive_tokens, negative_tokens, neutral_tokens

def evaluation(test_data,classifier):
    
    #accuracy=np.round(classify.accuracy(classifier, test_data),3)
    #print("Accuracy of Model is:", accuracy)
    #print(classifier.show_most_informative_features(10))
    
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
     
    for i, (feats, label) in enumerate(test_data):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    
    pos_precision=np.round(nltk.precision(refsets['Positive'], testsets['Positive']),3)
    pos_recall=np.round(nltk.recall(refsets['Positive'], testsets['Positive']),3)
    pos_fscore=np.round(nltk.f_measure(refsets['Positive'], testsets['Positive']),3)
    neg_precision=np.round(nltk.precision(refsets['Negative'], testsets['Negative']),3)
    neg_recall=np.round(nltk.recall(refsets['Negative'], testsets['Negative']),3)
    neg_fscore=np.round(nltk.f_measure(refsets['Negative'], testsets['Negative']),3)

    print ('pos precision:', pos_precision)
    print ('pos recall:', pos_recall)
    print ('pos F-score:', pos_fscore)
    print ('neg precision:', neg_precision)
    print ('neg recall:', neg_recall)
    print ('neg F-score:', neg_fscore)
    try: #Try loop added for neutrals
        neu_precision=np.round(nltk.precision(refsets['Neutral'], testsets['Neutral']),3)
        neu_recall=np.round(nltk.recall(refsets['Neutral'], testsets['Neutral']),3)
        neu_fscore=np.round(nltk.f_measure(refsets['Neutral'], testsets['Neutral']),3)    
        
        print ('neu precision:', neu_precision)
        print ('neu recall:', neu_recall)
        print ('neu F-score:', neu_fscore)    
    except:
        pass

def cross_val(test_data,train_data):
    dataset=test_data+train_data
    k_folds = 5
    subset_size = int(len(dataset)/k_folds)
    accuracy_list=[]
    for i in range(k_folds):
        testing = dataset[i*subset_size:][:subset_size]
        training = dataset[:i*subset_size] + dataset[(i+1)*subset_size:]
    
        classifier = NaiveBayesClassifier.train(training)
        accuracy=classify.accuracy(classifier, testing)
        accuracy_list.append(accuracy)      
    accuracy=np.round(np.mean(accuracy_list),3)
    std=np.round(np.std(accuracy_list),3)
    print('accuracy = '+str(accuracy)+'+/-'+str(std))
    
def naive_bayes_model(candidates,neutral):
    pos, neg=twitter_samp()
    print('########## - General Tweets - ############')
    neu=[]
    train_data,test_data2=pre_process2(pos,neg,neu)
    classifier = NaiveBayesClassifier.train(train_data)

    
    pos,neg,neu=pre_process_1(candidates)
    if neutral == 'no':
        neu=[] #Use empty list to remove neutrals    
    train_data,test_data=pre_process2(pos,neg,neu)

    evaluation(test_data,classifier)
    cross_val(train_data,test_data2)
    
    print('######### - Dataset Tweets - ##############')
    classifier = NaiveBayesClassifier.train(train_data) 
    evaluation(test_data,classifier)  
    cross_val(train_data,test_data)   
def main():
    #mongo_downloader() #Uncomment to download
    candidates='Trump and Biden' #'Biden' or 'Trump' #Change candidates
    neutral='yes' #'no' #Whether to include neutral tweets or not
    naive_bayes_model(candidates,neutral)

if __name__ == '__main__':
    main()
