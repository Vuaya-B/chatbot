import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words

# Load intents JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

tags = []
xy = []
all_words = []

# Preprocess data: tokenize and extract patterns and tags
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Ignore certain characters
ignore_words = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Custom Dataset Class
class ChatData(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
hidden_size = 8
input_size = len(X_train[0])
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# Prepare Dataset and DataLoader
dataset = ChatData()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.float().to(device)  # Ensure correct data type
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}")

print(f"final loss, loss={loss.item():.4f}")


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}


FILE = "data.pth"
torch.save(data,FILE)

print(f"training complete. file saved to {FILE}")



