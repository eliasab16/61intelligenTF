import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bow(tokenized_sentence, words_set):
    bag = np.zeros(len(words_set), dtype=np.float32)
    tokenized_sentence = [stem(word) for word in tokenized_sentence]

    for idx, w in enumerate(words_set):
        if w in tokenized_sentence:
            bag[idx] = 1

    return bag
