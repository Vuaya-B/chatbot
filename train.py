import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np



with open('intents.json','r') as f:
    intents = json.load(f)

# print(intents)

tags = []
xy = []
all_words= []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignore_words = ["?", "!", ".", ","]

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))

print(tags)

tags = sorted(set(tags))
print(tags)
# print(all_words)


X_train = []
y_train = []

for(pattern_sentence, tag) in xy:

    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)