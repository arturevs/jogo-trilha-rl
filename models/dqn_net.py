# Arquivo: models/dqn_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrilhaDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(TrilhaDQN, self).__init__()

        # Input: (3, 24) -> 72 features
        self.input_dim = input_shape[0] * input_shape[1]
        
        # --- Feature Extractor ---
        # Aumentei para 512 neurônios para maior capacidade de abstração
        # Adicionei LayerNorm para estabilizar o treinamento com rede mais profunda
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        
        self.fc3 = nn.Linear(512, 512)
        self.ln3 = nn.LayerNorm(512)

        # --- Dueling Architecture Streams ---
        
        # 1. Value Stream (V): Avalia quão bom é o estado, independente da ação
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 2. Advantage Stream (A): Avalia a vantagem relativa de cada ação
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        # Flatten da entrada: (Batch, 3, 24) -> (Batch, 72)
        x = x.view(x.size(0), -1) 
        
        # Feature extraction com normalização
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))

        # Separa nos dois fluxos
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combinação Dueling: Q(s,a) = V(s) + (A(s,a) - média(A(s,a)))
        # Isso garante estabilidade matemática e melhor identificação da melhor ação
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values