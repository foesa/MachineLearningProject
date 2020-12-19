"""
Author: Efeosa Eguavoen

Date Created: 16/12/2020

Function:
This script:
1: Gets preprocessed tweets from a text file(tweet_tokens.txt)
2: Gets the TF-IDF scores for the tokens in the data
3: Trains a SVC Classifier using the data

Each function has a docsting about its function
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
import pandas as pd


def get_dataset():
    """
    This function loads in the dataset from 'tweet_tokens.txt' and then puts the data into a dataframe
    :return: Dataframe with preprocessed tweets
    """
    with open('tweet_tokens.txt') as json_file:
        data = pd.DataFrame(json.load(json_file))
        return data


def tf_idf(dataset):
    """
    Gets the TF-IDF scores for the tweets in the dataframe to be used for Other Machine Learning Algorithms
    :param dataset: Dataframe containing preprocessed tweets
    :return: X: Scores, Y:Sentiment values
    """
    data = dataset["tokens"].values
    tfidf_converter = TfidfVectorizer(min_df=5, max_df=0.7)
    wordDoc = [" ".join(x) for x in data]
    X = tfidf_converter.fit_transform(wordDoc)
    y = dataset["sentiment"].values
    # df = pd.DataFrame(X[0].T.todense(), index=tfidf_converter.get_feature_names(), columns=["TF-IDF"])
    # df = df.sort_values('TF-IDF', ascending=False)
    # print(df.head())
    return X, y


def hyper_pick(X, y):
    """
    Select the best parameters and hyperparameters for the training data using GridSearchCV
    :param X: Tf-IDF scores of tweet data
    :param y:Sentiment values of the tweets
    :return: Outputs a graph of data
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    param_grid = [{'C': [0.01, 0.1, 1, 10, 100], 'gamma': [10, 1, 0.1, 0.01, 0.001],
                   'kernel': ['rbf', 'poly', 'sigmoid']}, {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100],
                                                           'gamma': [10, 1, 0.1, 0.01, 0.001]}]
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(grid.best_params_)


def svm(X, y):
    """
    Trains SVM Classifier
    :param X: TF-IDF Scores
    :param y:
    :return:
    """
    labels = ['Positive', 'Negative', 'Neutral']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = SVC(kernel='rbf', gamma=1, C=100)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    print('Confusion Matrix: ', confusion_matrix(y_test, predict, labels=labels))
    print('F1 Score: ', f1_score(y_test, predict, labels=labels, average='micro'))
    print('Precision Score: ', precision_score(y_test, predict, labels=labels, average='micro'))
    print('Accuracy Score: ', accuracy_score(y_test, predict))


def main():
    data = get_dataset()
    X, y = tf_idf(data)
    # hyper_pick(X, y)
    svm(X, y)

    return


if __name__ == '__main__':
    main()
