import json
import time

'''
This creates relevant training data adding the tag to each statement
for the decoder to train on to prevent key error and have relevant training
data that was grabbed from selected statements we already chose
'''
json_file = json.load(open('encoder.json'))
output_file_string = "new_training_text.txt"
output_file = open(output_file_string, "w+")

for item in json_file["intents"]:
    print(item["tag"])
    for el in item["responses"]:
        out = item["tag"] + " " + el.strip() + "\n"
        output_file.write(out)