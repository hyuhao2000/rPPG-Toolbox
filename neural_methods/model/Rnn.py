import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #cnn
        self.cnn1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.cnn2 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.rnn = nn.LSTM(int(input_size/3), hidden_size, num_layers, batch_first=True, bidirectional=True)
        #dropout
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size*2, 1)
    
    def forward(self, x):

        B ,T, N, A, C = x.shape
        
        x = x.squeeze(3)
        #change to B,T,N*C
        # x = x.view(B, T, N*C)
        # 第二第三维度交换
        x = x.permute(0, 2, 1, 3)
        out= self.cnn1(x)
        out = self.relu1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = out.squeeze(1)
        out, _ = self.rnn(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out
    
if __name__ == '__main__':
    model = RNN_model(378, 512, 2)
    # print(model)
    x = torch.randn(8, 180, 378)
    output = model(x)
    # print(output.shape)
