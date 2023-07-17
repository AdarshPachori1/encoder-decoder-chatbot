import random
import json
import pickle
import numpy as np
import re
import heapq

import nltk
from nltk.stem.snowball import SnowballStemmer


from tensorflow.keras.models import load_model


from lstm_model_interface import generate_response
import torch
import os
import torch.nn as nn

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


class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        
        # define embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # define lstm layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        
        
        # set class variables
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # define model layers
        self.fc = nn.Linear(hidden_dim, output_size)
    
    
    def forward(self, x, hidden):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """  
        batch_size = x.size(0)
        x=x.long()
        
        # embedding and lstm_out 
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # stack up lstm layers
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout, fc layer and final sigmoid layer
        out = self.fc(lstm_out)
        
        # reshaping out layer to batch_size * seq_length * output_size
        out = out.view(batch_size, -1, self.output_size)
        
        # return last batch
        out = out[:, -1]

        # return one batch of output word scores and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim
        weights = next(self.parameters()).data
        hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                    weights.new(self.n_layers, batch_size, self.hidden_dim).zero_()
                    )
        
        # initialize hidden state with zero weights, and move to GPU if available
        
        return hidden

def load_model(filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename)

trained_rnn = load_model('./save/trained_rnn2')



if __name__ == "__main__":
    print("starting chatbot")
    relevancy_dictionaries = vectorize_category(intents)

    while True:
        message = input("enter message: ")
        ints = predict_class(message)
        res = get_response(ints, intents)
        feed_in_response = generate_response_words_list(res, 3)
        print(feed_in_response)
        gen_length = 10
        response = [generate_response(trained_rnn, feed_in_response, gen_length)[0]]
        print(response)
        temp = " ".join(" ".join(response).split(" ")[len(feed_in_response):])
        print("response: ", temp)
        # print(" ".join(generate_response(trained_rnn, feed_in_response, gen_length)))
        # print(temp) 
