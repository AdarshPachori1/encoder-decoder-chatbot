import json

human1_text = "./human1.txt"
human2_text = "./human2.txt"


fh1 = open(human1_text, "r")
fh2 = open(human2_text, "r")

human1 = [line.strip('\n') for line in fh1.readlines()]
human2 = [line.strip('\n') for line in fh2.readlines()]

fh1.close()
fh2.close()

categories = {
    "greeting": ["hi,", "hi!", "how are you", "is anyone there?", "hello", "good day", "whats up", "hey!", "greetings"],
    "weather": ["weather", " sun ", "sunny ", " rain "," rainy "," raining"],
    "plans": [" week ", " weekend ", "monday", "tuesday", "wednesday", "thursday","friday","saturday","sunday",
                " plan "," plans ", " any plans ", "what are you doing"],
    "experience": ["try", "tried", "experience", "exciting"],

    "movies": ["movie", "movies", "avatar", "avengers", "science-fiction"],
    "cooking": ["cook", "cookies", "chocolate", "cookie", "eat", "food", "noodle", "soup", "pho", "ramen"],
    "holidays": ["christmas", "festive", "new year", "holiday", "holidays"],
}

organized = [
    {
        "tag":"greeting",
        "patterns": [],
        "responses": []
    },
    {
        "tag":"weather",
        "patterns": [],
        "responses": []
    },
    {
        "tag":"plans",
        "patterns": [],
        "responses": []
    },
    # {
    #     "tag":"misc",
    #     "patterns": [],
    #     "responses": []
    # },
    {
        "tag":"experience",
        "patterns": [],
        "responses": []
    },
    {
        "tag":"movies",
        "patterns": [],
        "responses": []
    },
    {
        "tag":"cooking",
        "patterns": [],
        "responses": []
    },
    {
        "tag":"holidays",
        "patterns": [],
        "responses": []
    },
]

def clean(arr):
    new = arr.copy()
    for str in arr:
        if len(str) < 2:
            new.remove(str)
    return new

for i in range(len(human1)):
    for tag in categories.keys():
        for phrase in categories[tag]:
            if phrase in human1[i].lower():
                for o in range(len(organized)):
                    if tag == organized[o]["tag"]:
                        organized[o]["responses"].append(human1[i])
                        organized[o]["responses"].append(human2[i])
                        break

            
for i in range(len(organized)-1):
    organized[i]["patterns"] = categories[organized[i]["tag"]]
    organized[i]["responses"] = list(set(organized[i]["responses"]))
    check = False
    new_responses = []
    for line in organized[i]["responses"]:
        line_split = line.replace('!', '!*').replace('?','?*').replace('.','.*')
        line_split = line_split.split('*')
        line_split_c = line_split.copy()
        for sentence in line_split:
            check = False
            for phrase in organized[i]["patterns"]:
                if phrase in sentence:
                    check = True
            if not check:
                line_split_c.remove(sentence)
        new_responses.append(" ".join(line_split_c))
    organized[i]["responses"] = clean(new_responses)


for i in range(3):
    print('--------------------------------------------------------------------------------')
    print("tag: ", organized[i]["tag"])
    print("patterns: \n", organized[i]["patterns"])
    print("responses: \n", organized[i]["responses"]) 

json_object = json.dumps({"intents": organized}, indent=1)
 
# Writing to sample.json
with open("encoder.json", "w") as outfile:
    outfile.write(json_object)             

print("done")