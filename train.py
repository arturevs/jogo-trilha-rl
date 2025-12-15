import math
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt

from envs.trilha_gym import TrilhaEnv
from models.dqn_net import TrilhaDQN

# --- HYPERPARAMETERS ---
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
MEMORY_SIZE = 50000
TARGET_UPDATE = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- AGENTES AUXILIARES ---


class RandomAgent:
    def act(self, state, mask, game_engine):
        valid_indices = np.where(mask == 1)[0]
        if len(valid_indices) == 0:
            return 0
        return int(np.random.choice(valid_indices))


class TrainedModelAgent:
    def __init__(self, model_path, env):
        self.net = TrilhaDQN(env.observation_space.shape, env.action_space.n).to(DEVICE)
        self.net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.net.eval()

    def act(self, state, mask, game_engine):
        valid_indices = np.where(mask == 1)[0]
        if len(valid_indices) == 0:
            return 0

        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = self.net(state_t)
            tensor_mask = torch.BoolTensor(mask).to(DEVICE)
            q_values[0, ~tensor_mask] = -float("inf")
            return q_values.max(1)[1].item()


# --- UTILS DE TREINO ---


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask):
        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class EarlyStopping:
    def __init__(self, patience=200, min_delta=0.5, start_from=1000):
        """
        Args:
            patience (int): Quantos episódios esperar sem melhora.
            min_delta (float): Mínima mudança para considerar melhora.
            start_from (int): Episódio a partir do qual o Early Stopping começa a contar.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.start_from = start_from
        self.best_reward = -float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_reward, model, path, episode):
        # Sempre salvamos o melhor modelo se houver melhora real,
        # independente do episódio (queremos o 'best_model' sempre).
        if current_reward > self.best_reward + self.min_delta:
            self.best_reward = current_reward
            self.counter = 0
            torch.save(model.state_dict(), path)
            return True  # Salvou novo melhor

        # Lógica de parada só ativa após o aquecimento (start_from)
        if episode < self.start_from:
            self.counter = 0  # Mantém resetado durante o aquecimento
            return False

        # Se não melhorou e já passou do aquecimento, conta paciência
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True

        return False


def select_action(state, mask, model, epsilon):
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) == 0:
        return 0

    if random.random() < epsilon:
        return int(np.random.choice(valid_indices))

    with torch.no_grad():
        q_values = model(state)
        tensor_mask = torch.BoolTensor(mask).to(DEVICE)
        q_values[0, ~tensor_mask] = -float("inf")
        return q_values.max(1)[1].item()


def train_phase(phase_name, opponent_agent, num_episodes, save_path, load_path=None):
    print(f"\n>>> INICIANDO FASE: {phase_name}")
    env = TrilhaEnv()
    env.set_opponent(opponent_agent)

    policy_net = TrilhaDQN(env.observation_space.shape, env.action_space.n).to(DEVICE)

    if load_path:
        print(f"Carregando pesos pré-treinados de {load_path}...")
        policy_net.load_state_dict(torch.load(load_path, map_location=DEVICE))

    target_net = TrilhaDQN(env.observation_space.shape, env.action_space.n).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    # Configura Early Stopping para ignorar os primeiros 1000 episódios
    early_stopping = EarlyStopping(patience=300, min_delta=1.0, start_from=1000)

    rewards_history = []
    avg_rewards_history = []

    epsilon_start = 1.0 if not load_path else 0.5
    epsilon_end = 0.05
    epsilon_decay = num_episodes * 0.4

    pbar = tqdm(range(num_episodes), desc=phase_name, unit="ep")

    for episode in pbar:
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(
            -1.0 * episode / epsilon_decay
        )

        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        total_reward = 0
        done = False

        while not done:
            mask = env.get_action_mask()
            action_idx = select_action(state, mask, policy_net, epsilon)

            next_state_np, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated

            next_state = torch.FloatTensor(next_state_np).unsqueeze(0).to(DEVICE)
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
                b_action = torch.LongTensor(batch_action).unsqueeze(1).to(DEVICE)
                b_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(DEVICE)
                b_next_state = torch.cat(batch_next_state)
                b_done = torch.FloatTensor(batch_done).unsqueeze(1).to(DEVICE)

                current_q = policy_net(b_state).gather(1, b_action)

                with torch.no_grad():
                    next_q_values = target_net(b_next_state)
                    max_next_q = next_q_values.max(1)[0].unsqueeze(1)
                    expected_q = b_reward + (GAMMA * max_next_q * (1 - b_done))

                loss = nn.MSELoss()(current_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(total_reward)
        avg_reward = (
            np.mean(rewards_history[-50:])
            if len(rewards_history) > 50
            else np.mean(rewards_history)
        )
        avg_rewards_history.append(avg_reward)

        # Passamos o episódio atual para o early_stopping checar o start_from
        saved = early_stopping(avg_reward, policy_net, save_path, episode)

        status = "💾 Saved" if saved else ""
        if episode < early_stopping.start_from:
            status += " (Warmup)"

        pbar.set_postfix(
            {
                "AvgR": f"{avg_reward:.1f}",
                "Best": f"{early_stopping.best_reward:.1f}",
                "St": status,
            }
        )

        if early_stopping.early_stop:
            print(f"\nEarly Stopping at episode {episode}!")
            break

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, alpha=0.3, color="gray", label="Raw")
    plt.plot(avg_rewards_history, color="blue", label="Avg")
    plt.title(f"Treino: {phase_name}")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.savefig(f"curve_{phase_name}.png")
    plt.close()

    print(f"Fase {phase_name} concluída. Melhor modelo salvo em: {save_path}")


def main():
    # FASE 1: VS RANDOM
    train_phase(
        phase_name="VS_RANDOM",
        opponent_agent=RandomAgent(),
        num_episodes=3000,
        save_path="model_vs_random.pth",
    )

    # FASE 2: VS EXPERT (Fine-tuning)
    expert_opponent = TrainedModelAgent(
        model_path="model_vs_random.pth", env=TrilhaEnv()
    )

    train_phase(
        phase_name="VS_EXPERT",
        opponent_agent=expert_opponent,
        num_episodes=5000,
        save_path="model_vs_expert.pth",
        load_path="model_vs_random.pth",
    )


if __name__ == "__main__":
    main()
