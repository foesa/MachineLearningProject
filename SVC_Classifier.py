import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_dataset():
    with open('tweet_tokens.txt') as json_file:
        data = pd.DataFrame(json.load(json_file))
        return data


def tf_idf(dataset):
    data = dataset["tokens"].values
    tfidf_converter = TfidfVectorizer(min_df=5, max_df=0.7)
    wordDoc = [" ".join(x) for x in data]
    X = tfidf_converter.fit_transform(wordDoc)
    y = dataset["sentiment"].values
    # df = pd.DataFrame(X[0].T.todense(), index=tfidf_converter.get_feature_names(), columns=["TF-IDF"])
    # df = df.sort_values('TF-IDF', ascending=False)
    # print(df.head())
    return X, y


def c_pick(X, y):
    c_vals = np.linspace(0.00001, 10, 25)
    mean_list = []
    std_list = []

    for i in c_vals:
        error_list = []
        print("i: ", i)
        for s in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model = SVC(C=i, kernel='linear', gamma='auto')
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            error_list.append(accuracy_score(y_test, pred))
            print("s: ", s)

        error_list = np.array(error_list)
        mean = error_list.mean()
        mean_list.append(mean)
        std = error_list.std()
        std_list.append(std)

    plt.clf()
    plt.errorbar(c_vals, mean_list, yerr=std_list)
    plt.xlabel('C Value')
    plt.ylabel('Mean Error')
    plt.title('C vs Mean Error')
    plt.show()


def svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = SVC(kernel='linear', gamma='auto')
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    print('Accuracy Score: ', accuracy_score(y_test, predict))

def main():
    data = get_dataset()
    X, y = tf_idf(data)
    c_pick(X, y)
    # svm(X, y)

    return


if __name__ == '__main__':
    main()