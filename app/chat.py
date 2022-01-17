import random
import json
import torch
from app.model import NeuralNet
from app.vtf_funcs import bow, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("app/cs61-chatbot-intents.json", "r") as json_data:
    intents = json.load(json_data)

FILE = "app/data.pth"
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

ERROR_THRESHOLD = 0.25


def classify(sentence):
    sentence = tokenize(sentence)
    user_input = bow(sentence, words_set)
    user_input = user_input.reshape(1, user_input.shape[0])
    user_input = torch.from_numpy(user_input).to(device)

    output = model(user_input)
    results = [
        [i, torch.sigmoid(torch.Tensor([[r]])).item()]
        for i, r in enumerate(output[0])
        if torch.sigmoid(torch.Tensor([[r]])).item() > ERROR_THRESHOLD
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((tags[r[0]], r[1]))

    return return_list


def get_response(sentence):
    global context
    global filter_active

    not_understand = [
        "Sorry, I didn't quite understand",
        "Sorry, could you try again?",
        "Apologies, I didn't get that. Try again maybe?",
        "I didn't get that. Please try again.",
    ]
    yes_commit = [
        "Okay, thanks. I will let staff know about it.",
        "Noted. Staff will take care of it. Anything else?",
        "Got it. I will notify staff. Thanks.",
        "Okay, a staff member will take a look and fix it for you.",
        "OK. If there's an error, we will fix it for you.",
    ]
    yes_late = [
        "It seems like you still have late days left. Please use those first, and then come back here if you run into more problems.",
        "I suggest that you use up your late hours - this is exactly what they are for. We are happy to help more if those also aren't enough.",
        "Okay. I can grant you 2 more late days. Hope this helps.",
        "Alright. Given your circumstances, you can use up to 6 more late days.",
    ]
    repeat = [
        "Why are you repeating yourself?",
        "Didn't you just say the same thing?",
        "You've said that already. Try something else.",
        "Don't repeat yourself...",
        "If you keep saying the same thing, I won't be able to help you..",
        "Please don't just say the same thing.",
    ]

    if filter_active:
        important_storage[context] = sentence
        filter_active = False
        if context == "extra hours":
            return random.choice(yes_late)
        elif context == "commit issue":
            return random.choice(yes_commit)

    results = classify(sentence)

    if results:
        while results:
            tag = results[0][0]
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    # set context for this intent if necessary
                    if "context_set" in intent:
                        if intent["context_set"] == context:
                            context = None
                            return random.choice(repeat)
                        else:
                            context = intent["context_set"]
                    # return random response from intent's set if it's not contextual, and if it's contextual and
                    # matches the user's urrent context
                    if not "context_filter" in intent or (
                        context != None
                        and "context_filter" in intent
                        and context in intent["context_filter"]
                    ):
                        if "context_filter" in intent:
                            filter_active = True
                            return random.choice(intent["responses"][context])
                        else:
                            # a random response from the intent
                            return random.choice(intent["responses"])

            results.pop(0)

    context = None
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