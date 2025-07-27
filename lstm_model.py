import torch
import torch.nn as nn

class SMILESLstm(nn.Module):
    def __init__(self,vocab_size,embedding_dim=128,hidden_dim = 256,num_layers = 2,output_dim = 12,dropout = 0.3):
        super(SMILESLstm, self).__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim,padding_idx = 0)

        # lstm layer
        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            dropout = dropout,
            batch_first = True,
            bidirectional = True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        embedded = self.embeddings(x)
        lstm_out,(hn,cn) = self.lstm(embedded)
        last_hidden = hn[-1]
        dropped = self.dropout(last_hidden)
        output = self.fc(dropped)
        return output

