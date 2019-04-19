from utils import *
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Bidirectional
from keras.models import Model
from keras.initializers import Constant
from keras.utils.np_utils import to_categorical
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from keras import models
from sklearn.metrics import classification_report
from keras.models import load_model
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = "7"
# session = tf.Session(config=config)
session = tf.Session()
from keras.backend.tensorflow_backend import set_session
set_session(session)

max_features = 10000

maxlen = 300
batch_size = 32
nb_classes = 2

def main():
    EMBEDDING_DIM = 300
    print("Taking input")
    dates, labels, news = input_data2()
    # dates2, labels2, news2 = input_data3()
    # sentiment_news = sentiment_scores(news2)

    pretrained_embedding_fname = 'glove.6B.300d.txt'
    print("Creating word vector index mapping")
    embedding_index = mapping(pretrained_embedding_fname)

    sequences, word_index, padded = tokenize_and_pad(news, maxlen = maxlen)

    # split the data into a training set and a validation set
    # print(padded.shape)
    # sentiment_news = np.zeros(padded.shape)
    # padded = np.concatenate([padded,sentiment_news], 1)
    # EMBEDDING_DIM += 25
    # print(padded.shape)
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.2
    indices = np.arange(padded.shape[0])
    np.random.shuffle(indices)
    padded = padded[indices]
    labels = np.asarray(labels)
    labels = labels[indices]
    
    num_validation_samples = int(VALIDATION_SPLIT * padded.shape[0])
    num_test_samples = int(TEST_SPLIT * padded.shape[0])
    X_train = padded[:-(num_validation_samples+num_test_samples)]
    Y_train = labels[:-(num_validation_samples+num_test_samples)]
    X_test = padded[-(num_validation_samples+num_test_samples):-num_validation_samples]
    Y_test = labels[-(num_validation_samples+num_test_samples):-num_validation_samples]
    X_val = padded[-num_validation_samples:]
    Y_val = labels[-num_validation_samples:]
    
    Y_train = to_categorical(Y_train, nb_classes)
    Y_test = to_categorical(Y_test, nb_classes)
    Y_val = to_categorical(Y_val, nb_classes)

    print(X_train)
    # preparing embedding matrix
    embedding_matrix, num_words = build_embedding_matrix(word_index, EMBEDDING_DIM, embedding_index)

    for word,i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = models.Sequential()
    model.add(Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), dropout=0.2))
    model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2)) 
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    
    # model = load_model('Bi_LSTM_2.h5')
    model.fit(X_train, Y_train, batch_size=32, epochs=5, validation_data=(X_val, Y_val))

    pred_labels = model.predict(X_test, batch_size = 32)

    y_true = []
    y_pred = []
    for y in Y_test:
        y_true.append(np.argmax(y))

    for l in pred_labels:
        y_pred.append(np.argmax(l))

    # print(y_true)
    # print(y_pred)
    print("Accuracy : " + str(round(accuracy_score(y_true, y_pred)*100, 2)) + "%")
    print(classification_report(y_true,y_pred))
    # model.save('Bi_LSTM_2.h5')

if __name__ == '__main__':
    main()
