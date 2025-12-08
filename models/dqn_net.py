import torch.nn as nn
import torch.nn.functional as F

class TrilhaDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(TrilhaDQN, self).__init__()

        self.input_dim = input_shape[0] * input_shape[1]
        
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.fc4(x)