import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def input_data():
    df = pd.read_csv('data/Combined_News_DJIA.csv', parse_dates=True, index_col=0)

    dates = []
    news = []
    labels = []
    for index, row in df.iterrows():
        # labels.append(row['Label'])
        # dates.append(row['Date'])
        # news_row = ""
        for i in range(10):
            news.append(str(row['Top'+str(i+1)])[2:-1])
            labels.append(row['Label'])
            dates.append(row['Date'])
        # news.append(news_row)
    
    return dates, labels, news 

def input_data2():
    df = pd.read_csv('data/Combined_News_DJIA.csv', parse_dates=True, index_col=0)

    dates = []
    news = []
    labels = []
    for index, row in df.iterrows():
        labels.append(row['Label'])
        dates.append(row['Date'])
        news_row = ""
        for i in range(25):
            news_row += ' ' + (str(row['Top'+str(i+1)])[2:-1])
            # labels.append(row['Label'])
            # dates.append(row['Date'])
        news.append(news_row)
    
    return dates, labels, news 

def input_data3():
    df = pd.read_csv('data/Combined_News_DJIA.csv', parse_dates=True, index_col=0)

    dates = []
    news = []
    labels = []
    for index, row in df.iterrows():
        labels.append(row['Label'])
        dates.append(row['Date'])
        news_row = []
        for i in range(25):
            news_row.append( (str(row['Top'+str(i+1)])[2:-1]) )
            # labels.append(row['Label'])
            # dates.append(row['Date'])
        news.append(news_row)
    
    return dates, labels, news 

def sentiment_scores(news): 
    sentiment_news = []
    f = open('data/sentiment', 'w+')
    for row in news:
        temp_sent = []
        for sentence in row:
            sid_obj = SentimentIntensityAnalyzer()  
            sentiment_dict = sid_obj.polarity_scores(sentence) 
            if sentiment_dict['compound'] >= 0.05:
                temp_sent.append(1)
                f.write(str(1) + ' ')
            elif sentiment_dict['compound'] <= -0.05:
                temp_sent.append(-1)
                f.write(str(-1) + ' ')
            else:
                temp_sent.append(0)
                f.write(str(0) + ' ')
        f.write('\n')
        print(news.index(row), " " , temp_sent)
        sentiment_news.append(temp_sent)

    return sentiment_news

# The mapping function returns a mapping from words to their embedding vector in the form a dictionary.
def mapping(name):
    embedding_index = {}
    f = open('data/'+name, encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector
    f.close()
    return embedding_index

def tokenize_and_pad(lines, maxlen):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    word_index = tokenizer.word_index
    print("%s unique tokens found." % len(word_index))
    padded = pad_sequences(sequences, padding='pre', maxlen=maxlen)
    return sequences, word_index, padded

# Returns an embedding matrix of all given words linked to their token ID's
def build_embedding_matrix(word_index, Embedding_Dim, embedding_index):
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, Embedding_Dim))

    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, num_words