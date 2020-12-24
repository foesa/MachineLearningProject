"""TODO
1: Get Tweet Data
2: Turn into word vectors using Word Embedding(Will need to process the dataset somewhat for this)
3: Use the word vectors in CNN and RNN
4: Get Results
"""
from tensorflow.python.keras.layers import LSTM

from SVC_Classifier import get_dataset
from sklearn.model_selection import train_test_split
from gensim import models
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, \
    Embedding
from tensorflow.keras.models import Model, Sequential
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.model_selection import KFold


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.rc('font', size=18)
    plt.clf()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def reprocess_data(dataset):
    dataset['Text_Final'] = [' '.join(x) for x in dataset['tokens']]
    pos = []
    neg = []
    neu = []
    for l in dataset.sentiment:
        if l == 'Positive':
            pos.append(1)
            neg.append(0)
            neu.append(0)
        elif l == 'Negative':
            pos.append(0)
            neg.append(1)
            neu.append(0)
        else:
            pos.append(0)
            neg.append(0)
            neu.append(1)

    dataset['Pos'] = pos
    dataset['Neg'] = neg
    dataset['Neu'] = neu


def test_train_split(dataset):
    train_data, test_data = train_test_split(dataset, test_size=0.2)
    all_training_words = [word for tokens in train_data["tokens"] for word in tokens]
    training_sentence_lengths = [len(tokens) for tokens in train_data["tokens"]]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))
    print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
    print("Max sentence length is %s" % max(training_sentence_lengths))

    all_test_words = [word for tokens in test_data["tokens"] for word in tokens]
    test_sentence_lengths = [len(tokens) for tokens in test_data["tokens"]]
    TEST_VOCAB = sorted(list(set(all_test_words)))
    print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
    print("Max sentence length is %s" % max(test_sentence_lengths))

    return train_data, test_data, TRAINING_VOCAB, TEST_VOCAB


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors,
                                                                               generate_missing=generate_missing))
    return list(embeddings)


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    embedding_layer = Embedding(num_words, embedding_dim, weights=[embeddings], input_length=max_sequence_length,
                                trainable=False)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2, 3, 4, 5, 6]
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model


def RNN(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, weights=[embeddings], input_length=max_sequence_length,
                        trainable=False))
    model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(labels_index, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def cross_val_NN(k, X, y, train_embedding_weights, MAX_SEQUENCE_LENGTH,
                 train_word_index, EMBEDDING_DIM, label_names, max_epoch,
                 nnmodel):
    """
    Inputs: kfold number, data to evaluate X,y and setting for the model
    
    Outputs: The accuracy, precision, recall and standard deviation of the model
    """

    accuracy_list = []
    recall_list = []
    precision_list = []
    kf = KFold(n_splits=k)

    for train, test in kf.split(X, y):
        if nnmodel == 'RNN':
            model = RNN(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index) + 1, EMBEDDING_DIM,
                        len(list(label_names)))
        if nnmodel == 'CNN':
            model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index) + 1, EMBEDDING_DIM,
                            len(list(label_names)))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

        model.fit(X[train], y[train], epochs=max_epoch, batch_size=34, validation_data=(X[test], y[test]))
        # predict=model.predict(X[test])
        loss, accuracy, f1_score, precision, recall = model.evaluate(X[test], y[test], verbose=0)

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


def main():
    dataset = get_dataset()
    reprocess_data(dataset)
    train_data, test_data, TRAINING_VOCAB, TEST_VOCAB = test_train_split(dataset)
    word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
    word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    training_embeddings = get_word2vec_embeddings(word2vec, train_data, generate_missing=True)
    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_DIM = 300
    tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(train_data["Text_Final"].tolist())
    training_sequences = tokenizer.texts_to_sequences(train_data["Text_Final"].tolist())

    train_word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(train_word_index))
    train_embedding_weights = np.zeros((len(train_word_index) + 1, EMBEDDING_DIM))
    for word, index in train_word_index.items():
        train_embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)

    train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test_sequences = tokenizer.texts_to_sequences(test_data["Text_Final"].tolist())
    test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    label_names = ['Pos', 'Neg', 'Neu']
    y_train = train_data[label_names].values
    y_test = test_data[label_names].values
    x_train = train_cnn_data
    x_test = test_cnn_data
    # model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index) + 1, EMBEDDING_DIM,
    # len(list(label_names)))
    model = RNN(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index) + 1, EMBEDDING_DIM,
                len(list(label_names)))
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
    plot_history(history)
    '''
    X=np.concatenate((x_train,x_test))
    y=np.concatenate((y_train,y_test))    

    cross_val_NN(5,X,y,train_embedding_weights,MAX_SEQUENCE_LENGTH,
                 train_word_index,EMBEDDING_DIM,label_names,max_epoch=3,
                 nnmodel='RNN') 
                 #For CNN use nnmodel='CNN', max_epoch=2
    '''


if __name__ == '__main__':
    main()
