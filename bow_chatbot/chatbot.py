import random
import json
import pickle
import numpy as np
import re
import heapq

import nltk
from nltk.stem.snowball import SnowballStemmer

from tensorflow.keras.models import load_model

stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'} 

lemmatizer = SnowballStemmer(language='english')
intents = json.loads(open('encoder.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = sorted(pickle.load(open('classes.pkl', 'rb')))
# model = load_model('chatbotModel.h5')
model = load_model('chatbot_model.model')

def clean_up_setnence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.stem(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_setnence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    

    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        # print(i)
        if i['tag'] == tag:
            # print("tag: ", tag)
            result = random.choice(i['responses'])
            break
    # return result
    # to pass to decoder, pass a few words instead of hardcoding response
    out = re.sub(r'[^a-zA-Z0-9 ]', '', result).split()
    return out
    # high = min(3, len(out))
    # return out[0:high]

'''
pass in the intents data and return a dictionary for each category with responses
that are vectorized and have a count for each lemmatized word to determine relevancy
in a dictionary

return dictionary with key being tag and value being dictionary with key being word and value
    being relevancy
'''
def vectorize_category(json_data):
    out = {}
    for item in json_data["intents"]:
        ins = {}
        for el in item["responses"]:
            for word in el.split():
                if word.lower() not in stopwords:
                    temp = lemmatizer.stem(re.sub(r'[^a-zA-Z0-9 ]', '', word)).lower()
                    ins[temp] = ins[temp] + 1 if temp in ins.keys() else 1
        out[item["tag"]] = ins
    return out
        
'''
generate_response_words_list finds the most relative words in the chosen
responses words and returns the n most relevant words as well as the category

return list of relevant words with category name
'''
def generate_response_words_list(res, n):
    input_heap = []
    comp = []
    cur = set([])
    for item in res:
        if lemmatizer.stem(item).lower() not in cur and item.lower() not in stopwords:
            heapq.heappush(input_heap, (relevancy_dictionaries[ints[0]["intent"]][lemmatizer.stem(item).lower()], item.lower()))
            heapq.heappush(comp, (relevancy_dictionaries[ints[0]["intent"]][lemmatizer.stem(item).lower()], item.lower()))
            cur.add(lemmatizer.stem(item).lower())
            if len(input_heap) > n:
                # print(cur, input_heap)
                if input_heap[0][1] in cur:
                    cur.remove(input_heap[0][1])
                heapq.heappop(input_heap)
    feed_in_response = [item[1] for item in input_heap]
    feed_in_response.append(ints[0]["intent"])
    return feed_in_response

if __name__ == "__main__":
    print("starting chatbot")
    relevancy_dictionaries = vectorize_category(intents)

    while True:
        message = input("enter message: ")
        ints = predict_class(message)
        res = get_response(ints, intents)
        feed_in_response = generate_response_words_list(res, 3)
        print(feed_in_response)
