import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.nn as nn

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

def generate(rnn, prime_words, int_to_vocab, vocab_to_int, token_dict, sequence_length, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()

    prime_ids = [vocab_to_int[prime] for prime in prime_words]
    pad_word = '<PAD>'
    pad_value = vocab_to_int[pad_word]

    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)

    # prime_id = [prime_id]
    for i in range(len(prime_ids),0,-1):
        current_seq[-1][-i] = prime_ids[-i]

    # current_seq[-1][-1] = prime_id[0]
    # print(current_seq)
    predicted = [int_to_vocab[str(prime)] for prime in prime_ids]
    
    for _ in range(predict_len):
        # if train_on_gpu:
        #     current_seq = torch.LongTensor(current_seq).cuda()
        # else:
        #     current_seq = torch.LongTensor(current_seq)
        current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        # if(train_on_gpu):
        #     p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[str(word_i)]
        predicted.append(word)     
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq.cpu(), -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences

'''
    trained RNN is trained model
    prime_words is list of words that we want to use to start generation
    gen_length is how long response will be
'''
def generate_response(trained_rnn, prime_words, gen_length):
    generated_script = generate(trained_rnn, prime_words, int_to_vocab, vocab_to_int, token_dict, sequence_length=10, predict_len=gen_length)
    return generated_script.split('\n')


with open('./int_to_vocab2.json') as json_file:
    int_to_vocab = json.load(json_file)
with open('./vocab_to_int2.json') as json_file:
    vocab_to_int = json.load(json_file)
with open('./token_dict2.json') as json_file:
    token_dict = json.load(json_file)