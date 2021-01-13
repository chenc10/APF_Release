import torch.nn as nn
import torch.nn.functional as F

class LSTM_KWS(nn.Module):

    def __init__(self):
        super(LSTM_KWS, self).__init__()
        self.rnn = nn.LSTM(
            # input_size=28,
            input_size=10,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        self.out = nn.Linear(64,10)

    def forward(self,x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:,-1,:])
        return out
