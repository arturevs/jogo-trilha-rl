# Arquivo: envs/trilha_gym.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict

# Importação relativa (assumindo que estão na mesma pasta 'envs')
from .game_logic import TrilhaGame


class TrilhaEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self):
        super(TrilhaEnv, self).__init__()

        self.game = TrilhaGame()

        # AÇÃO (Action Space): Discreto com 120 possibilidades
        # 0-23: Colocar (Place)
        # 24-119: Mover (Move) -> Mapeados por (Origem * 4) + Direção
        self.action_space = spaces.Discrete(120)

        # OBSERVAÇÃO (Observation Space): Tensor 3x24
        # Canal 0: Minhas peças (1 se sim, 0 se não)
        # Canal 1: Peças inimigas
        # Canal 2: Espaços vazios (ou feature de 'pode colocar aqui')
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, 24), dtype=np.float32
        )

        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step = 0  # Reseta o contador
        self.game.reset()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action_idx: int):
        # --- CORREÇÃO: Captura quem está jogando ANTES de qualquer mudança ---
        player_who_moved = self.game.turn

        terminated = False
        truncated = False
        reward = 0
        info = {}

        self.current_step += 1

        try:
            # 1. Executa Ação
            if action_idx < 24:
                self.game.apply_place(action_idx)
                reward += 0.2
            else:
                move_idx = action_idx - 24
                start_pos = move_idx // 4
                direction_code = move_idx % 4
                dirs = ["d", "e", "c", "b"]

                if start_pos in self.game.ADJACENCY:
                    target_pos = self.game.ADJACENCY[start_pos].get(
                        dirs[direction_code]
                    )
                    if target_pos is not None:
                        self.game.apply_move(start_pos, target_pos)
                        reward += 0.2
                    else:
                        reward -= 10
                else:
                    reward -= 10

            # 2. Auto-Remoção
            if self.game.pending_removal:
                reward += 5.0
                removed = self._auto_remove_piece()

                if not removed:
                    self.game.pending_removal = False
                    self.game._switch_turn_logic()

            # 3. Checa Vitória
            winner = self.game.check_winner()
            if winner:
                terminated = True
                # --- CORREÇÃO: Usamos a variável capturada no início ---
                if winner == player_who_moved:
                    reward += 50
                else:
                    reward -= 50

            # 4. Checa Limite de Passos
            if self.current_step >= self.max_steps:
                truncated = True

        except ValueError:
            reward = -10

        done = terminated or truncated

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        obs = np.zeros((3, 24), dtype=np.float32)
        curr = self.game.turn
        opp = "R" if curr == "V" else "V"

        for i, p in enumerate(self.game.board):
            if p == curr:
                obs[0][i] = 1
            elif p == opp:
                obs[1][i] = 1
            else:
                obs[2][i] = 1
        return obs

    def _get_info(self):
        return {"turn": self.game.turn, "phase": self.game.phase}

    def _auto_remove_piece(self):
        """Tenta remover peça. Retorna True se conseguiu, False se falhou."""
        opponent = "R" if self.game.turn == "V" else "V"
        for i, piece in enumerate(self.game.board):
            if piece == opponent:
                try:
                    self.game.apply_remove(i)
                    return True  # Sucesso!
                except ValueError:
                    continue  # Tenta a próxima
        return False  # Falhou em remover qualquer peça

    def get_action_mask(self):
        # ... (Mantenha sua lógica de máscara igual) ...
        # Apenas copiei o cabeçalho para lembrar que ela existe.
        # Use o código que você já tinha aqui.
        mask = np.zeros(120, dtype=np.int8)
        # ... (Cole sua lógica aqui) ...

        # Re-inserindo a lógica que você já tinha para garantir:
        if self.game.phase == "PLACEMENT":
            for i in range(24):
                if self.game.is_valid_place(i):
                    mask[i] = 1
        elif self.game.phase == "MOVEMENT":
            dirs = ["d", "e", "c", "b"]
            current_player = self.game.turn
            for pos in range(24):
                if self.game.board[pos] == current_player:
                    for i, d in enumerate(dirs):
                        if pos in self.game.ADJACENCY:
                            target = self.game.ADJACENCY[pos].get(d)
                            if target is not None and self.game.board[target] is None:
                                mask[24 + (pos * 4) + i] = 1
        return mask
