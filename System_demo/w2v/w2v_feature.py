import gensim.downloader as api
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator


def initialize_w2v():
    w2v_model = api.load('word2vec-google-news-300')
    return w2v_model


class W2VFeature(BaseEstimator):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def transform(self, sentences):
        sentences['tokens'] = self.get_list_of_tokens(sentences)
        
        return self.to_w2v_matrix(sentences, self.model)
        
    def get_list_of_tokens(self, df_texts):
        stop_words = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = []
        texts = df_texts["sentence"].values
        for i in range(len(texts)):
            row = texts[i]
            # remove punctuation
            for ch in string.punctuation:
                row = row.replace(ch, " ")
            row = row.replace("   ", " ")
            row = row.replace("  ", " ")
            temp_line = []
            # remove stop words
            for word in row.split():
                if word not in stop_words:
                    temp_line.append(word)
            row = ' '.join(temp_line)
            # lemmatization
            temp_line = []
            for word in row.split():
                temp_line.append(wordnet_lemmatizer.lemmatize(word))
            tokens.append(temp_line)
        return tokens

    def create_sentence_embeddings(self, model, words_list):
        sentence_embedding = []
        for word in words_list:
            try:
                sentence_embedding.append(model[word])
            except KeyError:
                continue
#                 print(word + " is not in the vocabulary, skipping...")
        if len(sentence_embedding) == 0:
            sentence_embedding.append(np.zeros(300))
        return np.array(sentence_embedding)

    def to_w2v_matrix(self, df_data, model):
        sent_embs = np.zeros([df_data.shape[0], 300 * 4], dtype='float32')
        for i in range(df_data.shape[0]):
            object_a_embedding = self.create_sentence_embeddings(model, df_data["object_a"][i].split()).mean(axis=0)
            object_b_embedding = self.create_sentence_embeddings(model, df_data["object_b"][i].split()).mean(axis=0)
            aspect_embedding = self.create_sentence_embeddings(model, df_data["aspect"][i].split()).mean(axis=0)
            sentence_embedding = self.create_sentence_embeddings(model, df_data["tokens"][i]).mean(axis=0)
            sent_embs[i, :] = np.concatenate((object_a_embedding, object_b_embedding, aspect_embedding, sentence_embedding), axis=0)
        return sent_embs