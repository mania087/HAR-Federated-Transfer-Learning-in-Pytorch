import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, data_size, n_classes):
        super(SimpleCNN, self).__init__()
        self.n_chan = data_size
        self.n_classes = n_classes

        # CNN layer
        self.conv1 = nn.Conv1d(self.n_chan, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        self.drop = nn.Dropout(p=0.6)
        self.pool = nn.MaxPool1d(kernel_size=2,stride=2)

        # Fully connected layer
        self.lin3 = nn.Linear(3968, 100)
        self.lin4 = nn.Linear(100, self.n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        a = torch.relu(self.conv1(x))
        a = torch.relu(self.conv2(a))
        a = self.drop(a)
        a = self.pool(a)
        a = a.view((batch_size, -1))
        a = torch.relu(self.lin3(a))
        a = torch.relu(self.lin4(a))

        return a
    
if __name__ == '__main__':
    # dummy forward
    model = SimpleCNN(9,6)
    test_input = torch.rand(2, 9, 128)
    print(model(test_input).size())