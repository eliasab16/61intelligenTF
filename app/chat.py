import random
import json
import torch
from model import NeuralNet
from vtf_funcs import bow, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("cs61-chatbot-intents.json", "r") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

context = None
# indicate whether we reach a filter state and need to store user input (like pset number and reason)
filter_active = False
important_storage = {}

# retrieve data and hyperparameters
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
words_set = data["words_set"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "intelligenTF"


def classify(sentence):
    sentence = tokenize(sentence)
    user_input = bow(sentence, words_set)
    user_input = user_input.reshape(1, user_input.shape[0])
    user_input = torch.from_numpy(user_input).to(device)

    output = model(user_input)
    results = [[i, r] for i, r in enumerate(output[0]) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((tags[r[0]], r[1]))

    return return_list


def get_response(sentence):
    global context
    global filter_active

    if filter_active:
        important_storage[context] = sentence
        filter_active = False
        return random.choice(
            [
                "Okay, thanks. I will let staff know about it.",
                "Noted. Staff will take care of it. Anything else?",
                "Got it. I will notify staff. Thanks.",
                "Okay, a staff member will take a look and fix it for you.",
                "OK. If there's an error, we will fix it for you.",
            ]
        )

    not_understand = [
        "Sorry, I didn't quite understand",
        "Sorry, could you try again?",
        "Apologies, I didn't get that. Try again maybe?",
    ]

    results = classify(sentence)

    if results:
        while results:
            tag = results[0][0]
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    # set context for this intent if necessary
                    if "context_set" in intent:
                        context = intent["context_set"]
                    # return random response from intent's set if it's not contextual, and if it's contextual and
                    # matches the user's urrent context
                    if not "context_filter" in intent or (
                        context != None
                        and "context_filter" in intent
                        and intent["context_filter"] == context
                    ):
                        if "context_filter" in intent:
                            filter_active = True
                        # a random response from the intent
                        return random.choice(intent["responses"])

            results.pop(0)

    return random.choice(not_understand)


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")

        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)