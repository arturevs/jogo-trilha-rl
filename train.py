# Arquivo: train.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque

# Nossas importações modulares
from envs.trilha_gym import TrilhaEnv
from models.dqn_net import TrilhaDQN

# --- HYPERPARAMETERS ---
BATCH_SIZE = 64
GAMMA = 0.99        # Valorização do futuro
EPSILON_START = 1.0 # Começa 100% aleatório
EPSILON_END = 0.05  # Termina 5% aleatório
EPSILON_DECAY = 1000
LR = 1e-3           # Taxa de aprendizado
MEMORY_SIZE = 10000
TARGET_UPDATE = 100 # Atualiza rede alvo a cada 100 episódios

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- REPLAY BUFFER (Memória) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask):
        # Guardamos também a máscara do próximo estado para calcular o Q-Target corretamente
        self.buffer.append((state, action, reward, next_state, done, mask))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# --- FUNÇÃO DE SELEÇÃO DE AÇÃO (Com Máscara) ---
def select_action(state, mask, model, epsilon):
    # state shape: (1, 3, 24)
    # mask shape: (120,)
    
    # Exploração (Random válido)
    if random.random() < epsilon:
        valid_indices = np.where(mask == 1)[0]
        if len(valid_indices) == 0:
            return 0 # Fallback (não deve acontecer se lógica estiver certa)
        return int(np.random.choice(valid_indices))
    
    # Exploração (Rede Neural)
    with torch.no_grad():
        q_values = model(state) # Tensor (1, 120)
        
        # TRUQUE DO MASCARAMENTO:
        # Define Q-values inválidos para -infinito antes de escolher o maior
        tensor_mask = torch.BoolTensor(mask).to(device)
        q_values[0, ~tensor_mask] = -float('inf')
        
        return q_values.max(1)[1].item() # Retorna o índice com maior valor

# --- LOOP PRINCIPAL ---
def train():
    env = TrilhaEnv()
    
    # Duas redes: Policy (treina agora) e Target (estável para cálculo de erro)
    policy_net = TrilhaDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = TrilhaDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    num_episodes = 5000

    print(f"Iniciando treino em: {device}")

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device) # Adiciona dimensão de batch (1, 3, 24)
        total_reward = 0
        done = False
        
        while not done:
            # 1. Obter Máscara Atual
            mask = env.get_action_mask()
            
            # 2. Escolher Ação
            action_idx = select_action(state, mask, policy_net, epsilon)
            
            # 3. Executar no Ambiente
            next_state_np, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated
            
            # Converter p/ Tensor
            next_state = torch.FloatTensor(next_state_np).unsqueeze(0).to(device)
            next_mask = env.get_action_mask() # Máscara do PRÓXIMO estado (importante p/ Bellman)
            
            # 4. Salvar na Memória
            memory.push(state, action_idx, reward, next_state, done, next_mask)
            
            state = next_state
            total_reward += reward
            
            # 5. Etapa de Aprendizado (Optimize)
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                # Transpor o batch (lista de tuplas -> tupla de listas)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done, batch_next_mask = zip(*transitions)
                
                b_state = torch.cat(batch_state)
                b_action = torch.LongTensor(batch_action).unsqueeze(1).to(device)
                b_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                b_next_state = torch.cat(batch_next_state)
                b_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
                # Nota: b_next_mask seria usado em Double DQN avançado, aqui vamos simplificar
                
                # Q(s, a) atuais
                current_q = policy_net(b_state).gather(1, b_action)
                
                # Q(s', a') ótimos futuros (Target)
                # Aqui o target não usa máscara, ele pega o max absoluto.
                # Se quisermos ser estritos, deveríamos aplicar máscara no next_state também.
                with torch.no_grad():
                    next_q_values = target_net(b_next_state)
                    max_next_q = next_q_values.max(1)[0].unsqueeze(1)
                    expected_q = b_reward + (GAMMA * max_next_q * (1 - b_done))
                
                # Loss e Backpropagation
                loss = nn.MSELoss()(current_q, expected_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decaimento de Epsilon
        epsilon = max(EPSILON_END, epsilon * 0.995)
        
        # Atualiza rede alvo
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episódio {episode}: Reward Total: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    # Salvar modelo final
    torch.save(policy_net.state_dict(), "trilha_final.pth")
    print("Treino concluído!")

if __name__ == "__main__":
    train()