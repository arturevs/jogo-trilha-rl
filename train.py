# Arquivo: train.py
import math
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt  # <--- Necessário: pip install matplotlib

from envs.trilha_gym import TrilhaEnv
from models.dqn_net import TrilhaDQN

# --- HYPERPARAMETERS ---
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 1000
LR = 1e-3
MEMORY_SIZE = 50000
TARGET_UPDATE = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask):
        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def select_action(state, mask, model, epsilon):
    # Verifica se existem ações válidas na máscara
    valid_indices = np.where(mask == 1)[0]

    # SE A MÁSCARA ESTIVER VAZIA (Agente Trancado), não tente prever.
    # Retorna uma ação aleatória qualquer apenas para o step() processar
    # e o check_winner() detectar a derrota logo em seguida.
    if len(valid_indices) == 0:
        return 0

    if random.random() < epsilon:
        return int(np.random.choice(valid_indices))

    with torch.no_grad():
        q_values = model(state)
        tensor_mask = torch.BoolTensor(mask).to(device)
        q_values[0, ~tensor_mask] = -float("inf")
        return q_values.max(1)[1].item()


def save_training_plot(rewards, avg_rewards, episode):
    """Gera e salva o gráfico de evolução do treinamento."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward (Episódio)", alpha=0.3, color="gray")
    plt.plot(avg_rewards, label="Média Móvel (50)", color="red", linewidth=2)

    plt.title(f"Treinamento Trilha RL - Episódio {episode}")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("training_curve.png")
    plt.close()  # Libera memória


def train():
    env = TrilhaEnv()

    policy_net = TrilhaDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = TrilhaDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    num_episodes = 2500

    # Histórico para métricas
    recent_rewards = deque(maxlen=50)
    history_rewards = []  # Histórico completo para o plot
    history_avg_rewards = []  # Histórico da média para o plot

    print(f"Iniciando treino em: {device}")
    pbar = tqdm(range(num_episodes), desc="Treinando", unit="ep")

    for episode in pbar:
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(
            -1.0 * episode / EPSILON_DECAY
        )

        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        total_reward = 0
        done = False

        while not done:
            mask = env.get_action_mask()
            action_idx = select_action(state, mask, policy_net, epsilon)

            next_state_np, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated

            next_state = torch.FloatTensor(next_state_np).unsqueeze(0).to(device)
            next_mask = env.get_action_mask()

            memory.push(state, action_idx, reward, next_state, done, next_mask)

            state = next_state
            total_reward += reward

            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                (
                    batch_state,
                    batch_action,
                    batch_reward,
                    batch_next_state,
                    batch_done,
                    _,
                ) = zip(*transitions)

                b_state = torch.cat(batch_state)
                b_action = torch.LongTensor(batch_action).unsqueeze(1).to(device)
                b_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                b_next_state = torch.cat(batch_next_state)
                b_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

                current_q = policy_net(b_state).gather(1, b_action)

                with torch.no_grad():
                    next_actions = policy_net(b_next_state).argmax(1).unsqueeze(1)
                    max_next_q = target_net(b_next_state).gather(1, next_actions)
                    expected_q = b_reward + (GAMMA * max_next_q * (1 - b_done))

                loss = nn.MSELoss()(current_q, expected_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Atualiza métricas
        recent_rewards.append(total_reward)
        avg_reward = sum(recent_rewards) / len(recent_rewards)

        # Guarda no histórico global para plotagem
        history_rewards.append(total_reward)
        history_avg_rewards.append(avg_reward)

        pbar.set_postfix({"Eps": f"{epsilon:.2f}", "AvgR": f"{avg_reward:.1f}"})

        # --- CHECKPOINT E LOGS ---
        if episode > 0 and episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), "checkpoint_trilha.pth")
            tqdm.write(f"💾 Checkpoint salvo no episódio {episode}")

        # --- PLOTAGEM GRÁFICA ---
        if episode > 0 and episode % 250 == 0:
            save_training_plot(history_rewards, history_avg_rewards, episode)
            tqdm.write("📈 Gráfico atualizado: training_curve.png")

    torch.save(policy_net.state_dict(), "trilha_final.pth")
    save_training_plot(history_rewards, history_avg_rewards, num_episodes)
    print("Treino concluído!")


if __name__ == "__main__":
    train()
