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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, \
    roc_curve, auc, classification_report, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from matplotlib.gridspec import GridSpec


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
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
    print('F1 Score: ', f1_score(y_test, predict, labels=labels, average='macro'))
    print('Precision Score: ', precision_score(y_test, predict, labels=labels, average='macro'))
    print('Accuracy Score: ', accuracy_score(y_test, predict))
    plot_multiclass_roc(clf, X_test, y_test, 3)
    print(classification_report(y_test, predict, digits=3))
    print('Recall: ', recall_score(y_test, predict, labels=labels, average='macro'))


def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6)):
    """
    Plots the multiclass ROC Curve for the classifier
    :param clf: Trained Classifier
    :param X_test: Test data
    :param y_test: Output labels for test data
    :param n_classes: Number of output classes
    :param figsize: Size of output graph
    """
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    plt.rcParams.update({'axes.titlesize': 'xx-large'})
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.set_title('Receiver operating characteristic example', fontsize=20)
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best", fontsize=12)
    ax.grid(alpha=.4)
    plt.show()


def cross_val(k, model, X, y):
    """
    Inputs: kfold number, machine learning model, data to evaluate X,y

    Outputs: The accuracy, recall and precision and their respective standard deviations for the model
    """
    accuracy_list = []
    recall_list = []
    precision_list = []
    labels = ['Positive', 'Negative', 'Neutral']
    kf = KFold(n_splits=k)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        predict = model.predict(X[test])

        accuracy = accuracy_score(y[test], predict)
        recall = recall_score(y[test], predict, labels=labels, average='macro')
        precision = precision_score(y[test], predict, labels=labels, average='macro')
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)

    accuracy_end = np.mean(accuracy_list)
    std = np.std(accuracy_list)
    recall = (np.mean(recall_list), np.std(recall_list))
    precision = (np.mean(precision_list), np.std(precision_list))

    print('Accuracy: ', accuracy_end)
    print('Standard Deviation', std)
    print('Recall', recall)
    print('Precision', precision)

    return accuracy_end, std, recall, precision


def plot_accuracy(X, y):
    """
    Purpose:
        Plots the accuracy and standard deviation for different gamma, C
        Using to fine tune these parameters
    """
    gamma = [0.1, 1, 10, 100]
    C = [0.1, 1, 10, 100]
    plotx = [0, 1, 0, 1]  # lists for plotting
    ploty = [0, 0, 1, 1]  # lists for plotting
    gs = GridSpec(2, 2, wspace=0.3, hspace=0.3)
    fig = plt.figure()
    plt.rc('font', size=18)
    l = 0
    for i in gamma:

        gx = plotx[l]
        gy = ploty[l]
        ax = fig.add_subplot(gs[gx, gy])
        accuracy_list = []
        std_list = []
        for c in C:
            model = SVC(kernel='rbf', gamma=i, C=c)
            accuracy, std, _, _ = cross_val(5, model, X, y)
            accuracy_list.append(accuracy)
            std_list.append(std)
        plt.errorbar(C, accuracy_list, yerr=std_list)
        ax.set_title('Gamma = ' + str(i))
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('C')
        plt.xscale('log')
        l = l + 1


def main():
    data = get_dataset()
    X, y = tf_idf(data)
    # hyper_pick(X, y)
    svm(X, y)
    cross_val(5, SVC(kernel='rbf', gamma=1, C=100), X, y)
    # plot_accuracy(X,y)
    return


if __name__ == '__main__':
    main()
