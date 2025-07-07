import torch.nn as nn
import torch
class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearLayer,self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class baseNet(nn.Module):
    def __init__(self, hidden_size_1=1000, hidden_size_2=2000):
        super(baseNet,self).__init__()
        self.linear1 = LinearLayer(100, hidden_size_1) 
        self.linear2 = LinearLayer(hidden_size_1, hidden_size_2) 
        self.linear3 = LinearLayer(hidden_size_2, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = baseNet().to(device)
    input = torch.randn((4, 100), device=device)
    print(model(input).shape)