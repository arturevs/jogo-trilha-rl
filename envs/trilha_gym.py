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

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.game.reset()

        # Se quiser inicialização aleatória ("Inteligente") para treinar Fase 2,
        # você chamaria sua função customizada aqui.

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action_idx: int):
        """Executa um passo no ambiente."""
        terminated = False
        truncated = False
        reward = 0
        info = {}

        try:
            # 1. Decodificar Ação
            if action_idx < 24:
                # Ação de PLACEMENT
                self.game.apply_place(action_idx)
                reward += 0.5  # Pequeno incentivo por jogar válido

            else:
                # Ação de MOVEMENT (Indices 24 a 119)
                move_idx = action_idx - 24
                start_pos = move_idx // 4
                direction_code = move_idx % 4  # 0:d, 1:e, 2:c, 3:b

                # Mapa de direções (precisa bater com game_logic)
                dirs = ["d", "e", "c", "b"]
                direction_char = dirs[direction_code]

                # Descobrir vizinho alvo
                if start_pos in self.game.ADJACENCY:
                    target_pos = self.game.ADJACENCY[start_pos].get(direction_char)

                    if target_pos is not None:
                        self.game.apply_move(start_pos, target_pos)
                        reward += 0.5
                    else:
                        reward -= 10  # Punição: tentou mover para fora do tabuleiro
                        # Em tese, o Masking impede isso, mas é bom ter salvaguarda.
                else:
                    reward -= 10  # Punição: Posição inválida

            # 2. Lógica de Remoção (Se aplicável)
            # Simplificação para RL: Se fez trilha, remove uma peça inimiga aleatória
            # para não explodir o espaço de ação com mais decisões agora.
            if self.game.pending_removal:
                reward += 5.0  # Grande recompensa por fazer trilha
                self._auto_remove_piece()

            # 3. Checar Fim de Jogo
            winner = self.game.check_winner()
            if winner:
                terminated = True
                if winner == self.game.turn:  # Cuidado: check_winner vê o estado atual
                    # Normalmente, quem jogou por último ganha
                    reward += 50
                else:
                    reward -= 50

        except ValueError:
            # Pegou movimento inválido (ex: colocar onde já tem peça)
            # O Action Masking deve prevenir isso, mas se falhar:
            reward = -10
            # Não termina o jogo, apenas pune e segue (ou perde a vez)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """Converte estado do jogo para Matriz (3, 24)"""
        obs = np.zeros((3, 24), dtype=np.float32)

        current_player = self.game.turn
        opponent = "R" if current_player == "V" else "V"

        for i, piece in enumerate(self.game.board):
            if piece == current_player:
                obs[0][i] = 1
            elif piece == opponent:
                obs[1][i] = 1
            else:
                obs[2][i] = 1  # Vazio

        return obs

    def _get_info(self):
        return {"turn": self.game.turn, "phase": self.game.phase}

    def _auto_remove_piece(self):
        """Remove a primeira peça válida do inimigo (Simplificação para v1)"""
        opponent = "R" if self.game.turn == "V" else "V"
        for i, piece in enumerate(self.game.board):
            if piece == opponent:
                # Tenta remover. Se for inválido (moinho fechado), try/except cuida
                try:
                    self.game.apply_remove(i)
                    break
                except ValueError:
                    continue

    def get_action_mask(self):
        """Copia da lógica de máscara que discutimos"""
        mask = np.zeros(120, dtype=np.int8)

        if self.game.phase == "PLACEMENT":
            for i in range(24):
                if self.game.is_valid_place(i):
                    mask[i] = 1

        elif self.game.phase == "MOVEMENT":
            dirs = ["d", "e", "c", "b"]
            current_player = self.game.turn

            for pos in range(24):
                if self.game.board[pos] == current_player:
                    # Checa as 4 direções
                    for i, d in enumerate(dirs):
                        if pos in self.game.ADJACENCY:
                            target = self.game.ADJACENCY[pos].get(d)
                            if target is not None and self.game.board[target] is None:
                                action_idx = 24 + (pos * 4) + i
                                mask[action_idx] = 1
        return mask
