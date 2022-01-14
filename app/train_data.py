import json
from vtf_funcs import tokenize, stem, bow
import numpy as np
from model import NeuralNet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open("app/cs61-chatbot-intents.json", "r") as intent_data:
    intents = json.load(intent_data)


def main():
    words_set = []
    tags = []
    docs = []
    ignore_words = ["!", ",", ".", "?", ";", ":"]

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)

        for pattern in intent["patterns"]:
            # tokenize and combine with words_set
            tokenized_words = tokenize(pattern)
            words_set.extend(stem(word) for word in tokenized_words)
            docs.append((tokenized_words, tag))

    # remove duplicates and sort
    words_set = sorted(set(words_set))
    tags = sorted(set(tags))

    x_train = []
    y_train = []

    # create a compatible training set (using 0,1 instead of words)
    for (pattern_sentence, tag) in docs:
        bag_of_words = bow(pattern_sentence, words_set)
        x_train.append(bag_of_words)
        y_train.append(tags.index(tag))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(x_train)
            self.x_data = x_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    # define hyperparameters for our models
    batch_size = 8
    hidden_size = 8
    input_size = len(x_train[0])
    output_size = len(tags)
    learning_rate = 0.001
    num_epochs = 1000

    dataset = ChatDataset()
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_size, hidden_size, output_size)

    # loss function and optimizer
    entropy = nn.CrossEntropyLoss()  # initialize internal module state
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            # push to the device
            words = words.to(device)
            labels = labels.to(device)

            # forward propagation
            outputs = model.forward(words)
            loss = entropy(
                outputs, labels
            )  # calculate the loss between outputs and actual labels

            # back propagation and optimization
            optimizer.zero_grad()  # sets the gradients the optimized tensors to zero
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch {epoch}/{num_epochs}, loss = {loss.item():.4f}")

    print(f"final loss, loss={loss.item():.4f}")

    # save the training data

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "words_set": words_set,
        "tags": tags,
    }

    # serialize and save to a pickled file
    FILE = "app/data.pth"
    torch.save(data, FILE)

    print(f"Training complete. File saved to {FILE}")


if __name__ == "__main__":
    main()