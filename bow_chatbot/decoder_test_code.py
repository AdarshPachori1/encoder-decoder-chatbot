from lstm_model_interface import generate_response
import torch
import os
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

trained_rnn = load_model('./save/trained_rnn')

# prime_words = ['what','are','you','doing','today']
# response = generate_response(trained_rnn, prime_words, 50)
# print(response)


while True:
    user = input('user: ')
    if (user.lower() == 'quit'):
        break
    response = generate_response(trained_rnn, user.split(),50)
    print('chatbot:',' '.join(response))
