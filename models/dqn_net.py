import torch.nn as nn
import torch.nn.functional as F

class TrilhaDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(TrilhaDQN, self).__init__()
        
        # input_shape será (3, 24) -> Flatten -> 72
        self.input_dim = input_shape[0] * input_shape[1]
        
        # Arquitetura: 72 -> 128 -> 256 -> 120
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_actions) # Saída: 120 Q-values

    def forward(self, x):
        # x chega como (Batch_Size, 3, 24)
        # Precisamos transformar em (Batch_Size, 72)
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Não usamos Softmax aqui porque DQN prevê Q-Values (recompensa esperada),
        # não probabilidades diretas. O valor pode ser qualquer número real.
        return self.fc3(x)