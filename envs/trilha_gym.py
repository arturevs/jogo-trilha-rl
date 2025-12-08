# Arquivo: envs/trilha_gym.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict

from .game_logic import TrilhaGame


class TrilhaEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self):
        super(TrilhaEnv, self).__init__()
        self.game = TrilhaGame()
        self.action_space = spaces.Discrete(120)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, 24), dtype=np.float32
        )

        # Limite de passos
        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.game.reset()
        return self._get_obs(), self._get_info()

    def step(self, action_idx: int):
        player_who_moved = self.game.turn

        terminated = False
        truncated = False
        reward = 0

        self.current_step += 1

        try:
            step_penalty = -0.01
            reward += step_penalty

            if action_idx < 24:
                self.game.apply_place(action_idx)
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
                    else:
                        reward -= 10  # Punição Ilegal mantém alta
                else:
                    reward -= 10

            # --- AUTO-REMOÇÃO ---
            if self.game.pending_removal:
                reward += 10.0  # Aumentei o incentivo para fazer trilhas
                removed = self._auto_remove_piece()
                if not removed:
                    self.game.pending_removal = False
                    self.game._switch_turn_logic()

            # --- VITÓRIA ---
            winner = self.game.check_winner()
            if winner:
                terminated = True
                if winner == player_who_moved:
                    reward += 100  # Aumentei para garantir que vale a pena
                else:
                    reward -= 100

        except ValueError:
            reward = -10  # Punição por tentar jogada impossível (regra do jogo)

        if self.current_step >= self.max_steps:
            truncated = True

        done = terminated or truncated

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ... (Mantenha _get_obs, _get_info, _auto_remove_piece e get_action_mask IGUAIS) ...
    # Vou repetir apenas para garantir que você tenha o arquivo completo se copiar/colar
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
        opponent = "R" if self.game.turn == "V" else "V"
        for i, piece in enumerate(self.game.board):
            if piece == opponent:
                try:
                    self.game.apply_remove(i)
                    return True
                except ValueError:
                    continue
        return False

    def get_action_mask(self):
        mask = np.zeros(120, dtype=np.int8)
        if self.game.phase == "PLACEMENT":
            for i in range(24):
                if self.game.is_valid_place(i):
                    mask[i] = 1
        elif self.game.phase == "MOVEMENT":
            dirs = ["d", "e", "c", "b"]
            curr = self.game.turn
            for pos in range(24):
                if self.game.board[pos] == curr:
                    for i, d in enumerate(dirs):
                        if pos in self.game.ADJACENCY:
                            target = self.game.ADJACENCY[pos].get(d)
                            if target is not None and self.game.board[target] is None:
                                mask[24 + (pos * 4) + i] = 1
        return mask
