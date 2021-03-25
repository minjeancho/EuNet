from torch import nn
import torch 

class MLP(nn.Module):
    def __init__(self, hyperparams, window_size = 4):
        super().__init__()
        hidden_sz = hyperparams["hidden_sz"]
        vocab_sz =  hyperparams["vocab_sz"]
        
        self.fc1 = nn.Linear(window_size, hidden_sz)
        self.fc2 = nn.Linear(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, vocab_sz)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.output(x)
        return x 
    

class LSTM(nn.Module):
    def __init__(self, hyperparams, window_size = 4):
        super().__init__()
        vocab_sz =  hyperparams["vocab_sz"]
        embed_sz = hyperparams["embed_sz"]
        rnn_sz =  hyperparams["rnn_sz"]
        
        self.embed = nn.Embedding(num_embeddings = vocab_sz, embedding_dim = embed_sz)
        self.lstm = nn.LSTM(embed_sz, rnn_sz, batch_first = True)
        self.output = nn.Linear(rnn_sz, vocab_sz) 
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embed(x)
        _, (x, _)  = self.lstm(x)
        x = self.tanh(x)
        x = self.output(x)
        return x 



class FeedbackModel(nn.Module):
    def __init__(self, hyperparams, window_size = 4):
        super().__init__()
        hidden_sz = hyperparams["hidden_sz"]
        vocab_sz =  hyperparams["vocab_sz"]
        
        self.fc1 = nn.Linear(window_size, hidden_sz)
        self.fc2 = nn.Linear(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, vocab_sz)
        self.memory_fc = nn.Linear(hidden_sz, hidden_sz)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x_in, x_memory):
        x0 = x_in
        x1 = self.sigmoid(self.fc1(x0))
        # add feedback 
        x1 = x1 + self.memory_fc(x_memory)
        x2 = self.sigmoid(self.fc2(x1))
        output = self.output(x2)
        return x2, output 


class EuNet(nn.Module):
    def __init__(self, hyperparams, window_size = 4):
        super().__init__()
        hidden_sz = hyperparams["hidden_sz"]
        vocab_sz =  hyperparams["vocab_sz"]
        G_sz = window_size + hidden_sz + hidden_sz + vocab_sz

        self.G0 = nn.Linear(G_sz, window_size)

        self.fc1 = nn.Linear(window_size, hidden_sz)
        self.G1 = nn.Linear(G_sz, hidden_sz)

        self.fc2 = nn.Linear(hidden_sz, hidden_sz)
        self.G2 = nn.Linear(G_sz,  hidden_sz)

        self.output = nn.Linear(hidden_sz, vocab_sz)
        self.G3 = nn.Linear(G_sz, vocab_sz)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x_in, x_all):
        x0 = x_in
        # add feedback 
        x0 = x0 + self.sigmoid(self.G0(x_all))
        x1 = self.sigmoid(self.fc1(x0))
        # add feedback 
        x1 = x1 + self.sigmoid(self.G1(x_all))
        x2 = self.sigmoid(self.fc2(x1))
        # add feedback 
        x2 = x2 + self.sigmoid(self.G2(x_all))
        output = self.output(x2)
        return x0, x1, x2, output 






