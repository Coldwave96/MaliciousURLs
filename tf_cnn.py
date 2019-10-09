# coding:utf-8
import os
import urllib.parse
import tensorflow as tf
import numpy as np
import tflearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

max_features = 20000

def get_query_list(filename):
    directory = str(os.getcwd()) + '\data\\train'
    filepath = directory + '\\' + filename
    data = open(filepath, 'r', encoding='utf-8').readlines()
    query_list = []
    for i in data:
        i = str(urllib.parse.unquote(i))
        query_list.append(i)
    return list(set(query_list))

def get_features_by_wordbag_tfidf():
    good_query_list = get_query_list('goodqueries.txt')
    bad_query_list = get_query_list('badqueries.txt')

    good_y = [0 for i in range(0, len(good_query_list))]
    bad_y = [1 for i in range(0, len(bad_query_list))]

    queries = good_query_list + bad_query_list
    y = good_y + bad_y

    vectorizer = CountVectorizer(decode_error='ignore', strip_accents='ascii', max_features=max_features, stop_words='english', max_df=1.0, min_df=1)

    x = vectorizer.fit_transform(queries)
    x = x.toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(x)
    x = tfidf.toarray()

    return x, y

def do_cnn_wordbad_tfidf(trainX, testX, trainY, testY):
    trainX = pad_sequences(trainX, value=0.)
    testX = pad_sequences(testX, value=0.)

    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    network = input_data(name='input')
    network = tflearn.embedding(network, input_dim=1000000, output_dim=128)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY, n_epoch=5, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=100, run_id="url")

if __name__ == "__main__":
    print("get_features_by_wordbag_tfidf")
    x, y = get_features_by_wordbag_tfidf()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    do_cnn_wordbag_tfidf(x_train, x_test, y_train, y_test)