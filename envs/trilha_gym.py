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

        self.max_steps = 200
        self.current_step = 0

        self.opponent_agent = None

    def set_opponent(self, agent):
        """Define quem jogará contra o modelo treinado (Random ou Frozen Model)."""
        self.opponent_agent = agent

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.game.reset()

        return self._get_obs(), self._get_info()

    def step(self, action_idx: int):
        """
        Executa:
        1. Movimento do Agente (V)
        2. Verifica Vitória/Trilha
        3. Se não acabou, Movimento do Oponente (R)
        4. Verifica Vitória/Trilha/Derrota
        5. Retorna estado e recompensa acumulada
        """
        self.current_step += 1
        total_reward = 0
        terminated = False
        truncated = False

        reward_agent, done_agent = self._execute_move(action_idx, player="V")
        total_reward += reward_agent

        if done_agent:
            return self._get_obs(), total_reward, True, False, self._get_info()

        if self.opponent_agent is not None:
            opp_state = self._get_obs()
            opp_mask = self.get_action_mask()
            opp_action = self.opponent_agent.act(opp_state, opp_mask, self.game)

            reward_opp, done_opp = self._execute_move(opp_action, player="R")

            total_reward -= reward_opp

            if done_opp:
                return self._get_obs(), total_reward, True, False, self._get_info()

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), total_reward, terminated, truncated, self._get_info()

    def _execute_move(self, action_idx, player):
        """Aplica o movimento e calcula recompensa pontual para AQUELE jogador."""
        reward = 0
        try:
            if action_idx < 24:
                self.game.apply_place(action_idx)
            else:
                move_idx = action_idx - 24
                start_pos = move_idx // 4
                direction_code = move_idx % 4
                dirs = ["d", "e", "c", "b"]
                if start_pos in self.game.ADJACENCY:
                    target = self.game.ADJACENCY[start_pos].get(dirs[direction_code])
                    if target is not None:
                        self.game.apply_move(start_pos, target)
                    else:
                        return -10, False
                else:
                    return -10, False

            if self.game.pending_removal:
                reward += 20.0
                self._auto_remove_piece()
                self.game.pending_removal = False
                self.game._switch_turn_logic()

            # --- VITÓRIA ---
            winner = self.game.check_winner()
            if winner:
                if winner == player:
                    reward += 100.0
                else:
                    reward -= 100.0
                return reward, True

        except ValueError:
            return -10, False

        return reward, False

    def _get_obs(self):
        obs = np.zeros((3, 24), dtype=np.float32)
        curr = self.game.turn  # Quem joga agora
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
        """Remove a primeira peça válida do oponente (simplificação para RL)."""
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
