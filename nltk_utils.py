import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

nltk.download('punkt_tab')
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenize_sentence, all_words):
        
        # sentence = ["hello", "how", "are", "you"]
        # words = ["hi","hello","I","you","bye","thanks","cool"]
        # bag   = [ 0,     1,    0,    1,    0,       0,     0]


        tokenize_sentence = [stem(w) for w in tokenize_sentence]

        bag = np.zeros(len(all_words), dtype=np.float32)


        for ind, w in enumerate(all_words):
        
             if w in tokenize_sentence:
                  bag[ind] = 1.0
        
        return bag


# a = "How long does shipping take?"

# print(a)
# a = tokenize(a)
# print(a)


# words = ["Organize","Organizing","Organizes"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)


# b = bag_of_words(sentence, words)
# print(b)