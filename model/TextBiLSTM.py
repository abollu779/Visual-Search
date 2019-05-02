import torch
import torch.nn as nn
from torch.nn.utils import rnn

class TextBiLSTM(nn.Module):
    def __init__(self):
        super(TextBiLSTM, self).__init__()

        self.in_size = 768 * 4
        self.num_layers = 2
        self.hidden_size = 512

        self.lstm = nn.LSTM(self.in_size, self.hidden_size, self.num_layers, bidirectional=True)

    def forward(self,x):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = rnn.pack_sequence(x).to(device)

        _,(h,c) = self.lstm(x,None)
 
        h = h.permute(1,0,2)
        h = h.contiguous().view(h.shape[0],-1)

        return h

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dut = TextBiLSTM().to(device)
    data = [torch.rand(5,3072),torch.rand(3,3072),torch.rand(2,3072)]
    out = dut(data)
    assert(out.shape==(3,2048))
    loss = sum(sum(out))
    loss.backward()
    print("Test passed!")
