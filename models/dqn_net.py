import torch.nn as nn
import torch.nn.functional as F


class TrilhaDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(TrilhaDQN, self).__init__()

        self.input_dim = input_shape[0] * input_shape[1]

        self.fc1 = nn.Linear(self.input_dim, 512)
        self.ln1 = nn.LayerNorm(512)

        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)

        self.fc3 = nn.Linear(512, 512)
        self.ln3 = nn.LayerNorm(512)

        self.value_stream = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
