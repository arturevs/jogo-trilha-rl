# Arquivo: envs/game_logic.py
class TrilhaGame:
    # --- DEFINIÇÕES ESTÁTICAS (Constantes) ---
    ADJACENCY = {
        0: {"d": 1, "b": 7},
        1: {"e": 0, "d": 2, "b": 9},
        2: {"e": 1, "b": 3},
        3: {"c": 2, "b": 4, "e": 11},
        4: {"c": 3, "e": 5},
        5: {"d": 4, "e": 6, "c": 13},
        6: {"d": 5, "c": 7},
        7: {"b": 6, "c": 0, "d": 15},
        8: {"d": 9, "b": 15},
        9: {"e": 8, "d": 10, "c": 1, "b": 17},
        10: {"e": 9, "b": 11},
        11: {"c": 10, "b": 12, "d": 3, "e": 19},
        12: {"c": 11, "e": 13},
        13: {"d": 12, "e": 14, "b": 5, "c": 21},
        14: {"d": 13, "c": 15},
        15: {"b": 14, "c": 8, "e": 7, "d": 23},
        16: {"d": 17, "b": 23},
        17: {"e": 16, "d": 18, "c": 9},
        18: {"e": 17, "b": 19},
        19: {"c": 18, "b": 20, "d": 11},
        20: {"c": 19, "e": 21},
        21: {"d": 20, "e": 22, "b": 13},
        22: {"d": 21, "c": 23},
        23: {"b": 22, "c": 16, "e": 15},
    }

    POSSIBLE_MILLS = [
        {0, 1, 2}, {2, 3, 4}, {4, 5, 6}, {6, 7, 0},
        {8, 9, 10}, {10, 11, 12}, {12, 13, 14}, {14, 15, 8},
        {16, 17, 18}, {18, 19, 20}, {20, 21, 22}, {22, 23, 16},
        {1, 9, 17}, {3, 11, 19}, {5, 13, 21}, {7, 15, 23},
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [None] * 24
        self.turn = "V"
        self.phase = "PLACEMENT"
        self.pieces_to_place = {"V": 9, "R": 9}
        self.pieces_on_board = {"V": 0, "R": 0}
        self.pending_removal = False
        self.history = []
        return self.get_state()

    def get_state(self):
        return {
            "board": self.board[:],
            "turn": self.turn,
            "phase": self.phase,
            "pending_removal": self.pending_removal,
            "counts": self.pieces_on_board.copy(),
        }

    # --- LÓGICA DE REGRAS (Validadores) ---

    def is_valid_place(self, pos_idx):
        if self.phase != "PLACEMENT":
            return False
        if self.board[pos_idx] is not None:
            return False
        if self.pending_removal:
            return False
        return True

    def is_valid_move(self, current_pos, target_pos):
        if self.phase != "MOVEMENT":
            return False
        if self.pending_removal:
            return False
        if self.board[current_pos] != self.turn:
            return False
        if self.board[target_pos] is not None:
            return False
        neighbors = self.ADJACENCY[current_pos].values()
        if target_pos not in neighbors:
            return False
        return True

    def is_valid_remove(self, pos_idx):
        """Checa se pode remover essa peça inimiga."""
        if not self.pending_removal:
            return False

        piece = self.board[pos_idx]
        opponent = "R" if self.turn == "V" else "V"

        if piece != opponent:
            return False

        # --- ALTERAÇÃO AQUI: Regra de proteção removida ---
        # Antes verificava is_part_of_mill, agora permite sempre.
        return True

    # --- AÇÕES (Executores) ---

    def apply_place(self, pos_idx):
        if not self.is_valid_place(pos_idx):
            raise ValueError(f"Movimento inválido: Place {pos_idx}")

        self.board[pos_idx] = self.turn
        self.pieces_to_place[self.turn] -= 1
        self.pieces_on_board[self.turn] += 1

        if self.check_mill_formed(pos_idx):
            self.pending_removal = True
        else:
            self._switch_turn_logic()

    def apply_move(self, start_pos, end_pos):
        if not self.is_valid_move(start_pos, end_pos):
            raise ValueError(f"Movimento inválido: Move {start_pos}->{end_pos}")

        self.board[start_pos] = None
        self.board[end_pos] = self.turn

        if self.check_mill_formed(end_pos):
            self.pending_removal = True
        else:
            self._switch_turn_logic()

    def apply_remove(self, pos_idx):
        if not self.is_valid_remove(pos_idx):
            raise ValueError("Remoção inválida")

        opponent = "R" if self.turn == "V" else "V"
        self.board[pos_idx] = None
        self.pieces_on_board[opponent] -= 1
        self.pending_removal = False

        self._switch_turn_logic()

    # --- AUXILIARES INTERNOS ---

    def check_mill_formed(self, pos_idx):
        player = self.board[pos_idx]
        for mill in self.POSSIBLE_MILLS:
            if pos_idx in mill:
                if all(self.board[p] == player for p in mill):
                    return True
        return False

    def is_part_of_mill(self, pos_idx, player):
        if self.board[pos_idx] != player:
            return False
        for mill in self.POSSIBLE_MILLS:
            if pos_idx in mill:
                if all(self.board[p] == player for p in mill):
                    return True
        return False

    def all_enemies_in_mill(self, player):
        # Mantido para compatibilidade, mas não é mais usado na validação
        indices_player = [i for i, p in enumerate(self.board) if p == player]
        if not indices_player:
            return False
        for idx in indices_player:
            if not self.is_part_of_mill(idx, player):
                return False
        return True

    def _switch_turn_logic(self):
        self.turn = "R" if self.turn == "V" else "V"
        if self.pieces_to_place["V"] == 0 and self.pieces_to_place["R"] == 0:
            self.phase = "MOVEMENT"

    def has_valid_moves(self, player):
        if self.phase == "PLACEMENT":
            return self.pieces_to_place[player] > 0
        if self.phase == "MOVEMENT":
            for pos, p in enumerate(self.board):
                if p == player:
                    if pos in self.ADJACENCY:
                        for neighbor in self.ADJACENCY[pos].values():
                            if self.board[neighbor] is None:
                                return True
            return False
        return True

    def check_winner(self):
        if self.phase == "MOVEMENT":
            if self.pieces_on_board["V"] < 3:
                return "R"
            if self.pieces_on_board["R"] < 3:
                return "V"
            if not self.has_valid_moves(self.turn):
                return "R" if self.turn == "V" else "V"
        return None