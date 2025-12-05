class TrilhaGame:
    # --- DEFINIÇÕES ESTÁTICAS (Constantes) ---
    # Mapeamento de vizinhos (Grafo)
    ADJACENCY = {
        # --- Anel Externo ---
        0: {'d': 1, 'b': 7},
        1: {'e': 0, 'd': 2, 'b': 9},
        2: {'e': 1, 'b': 3},
        3: {'c': 2, 'b': 4, 'e': 11},
        4: {'c': 3, 'e': 5},
        5: {'d': 4, 'e': 6, 'c': 13},
        6: {'d': 5, 'c': 7},
        7: {'b': 6, 'c': 0, 'd': 15},
        
        # --- Anel do Meio ---
        8: {'d': 9, 'b': 15},
        9: {'e': 8, 'd': 10, 'c': 1, 'b': 17},
        10: {'e': 9, 'b': 11},
        11: {'c': 10, 'b': 12, 'd': 3, 'e': 19},
        12: {'c': 11, 'e': 13},
        13: {'d': 12, 'e': 14, 'b': 5, 'c': 21},
        14: {'d': 13, 'c': 15},
        15: {'b': 14, 'c': 8, 'e': 7, 'd': 23},
        
        # --- Anel Interno ---
        16: {'d': 17, 'b': 23},
        17: {'e': 16, 'd': 18, 'c': 9},
        18: {'e': 17, 'b': 19},
        19: {'c': 18, 'b': 20, 'd': 11},
        20: {'c': 19, 'e': 21},
        21: {'d': 20, 'e': 22, 'b': 13},
        22: {'d': 21, 'c': 23},
        23: {'b': 22, 'c': 16, 'e': 15}
    }

    # Todas as trilhas possíveis (Linhas de vitória)
    POSSIBLE_MILLS = [
        {0, 1, 2},
        {2, 3, 4},
        {4, 5, 6},
        {6, 7, 0},
        {8, 9, 10},
        {10, 11, 12},
        {12, 13, 14},
        {14, 15, 8},
        {16, 17, 18},
        {18, 19, 20},
        {20, 21, 22},
        {22, 23, 16},
        {1, 9, 17},
        {3, 11, 19},
        {5, 13, 21},
        {7, 15, 23},
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        """Reinicia o jogo para o estado zero."""
        self.board = [None] * 24  # None, 'V', 'R'
        self.turn = "V"  # Quem joga agora
        self.phase = "PLACEMENT"  # PLACEMENT, MOVEMENT

        # Contadores
        self.pieces_to_place = {"V": 9, "R": 9}
        self.pieces_on_board = {"V": 0, "R": 0}

        # Estado de espera (quando faz trilha)
        self.pending_removal = False

        # Histórico simples para evitar loops (opcional, mas bom)
        self.history = []

        # Dicionário reverso para busca rápida (Onde está a peça X?)
        # Nota: Em RL puro, talvez não precisemos de nomes 'V1', 'V2'.
        # Apenas saber que tem um 'V' na casa 10 basta.
        # Vou manter simples: board[i] = 'V' ou 'R'.

        return self.get_state()

    def get_state(self):
        """Retorna uma cópia limpa do estado (para o Gym usar)."""
        return {
            "board": self.board[:],
            "turn": self.turn,
            "phase": self.phase,
            "pending_removal": self.pending_removal,
            "counts": self.pieces_on_board.copy(),
        }

    # --- LÓGICA DE REGRAS (Validadores) ---

    def is_valid_place(self, pos_idx):
        """Checa se pode colocar peça aqui."""
        if self.phase != "PLACEMENT":
            return False
        if self.board[pos_idx] is not None:
            return False
        if self.pending_removal:
            return False  # Se tem que remover, não pode colocar
        return True

    def is_valid_move(self, current_pos, target_pos):
        """Checa se pode mover daqui para lá."""
        if self.phase != "MOVEMENT":
            return False
        if self.pending_removal:
            return False

        # A peça na origem é minha?
        if self.board[current_pos] != self.turn:
            return False

        # O destino está vazio?
        if self.board[target_pos] is not None:
            return False

        # Eles são vizinhos? (Verifica no Grafo)
        # Forma simplificada de checar vizinhança sem direção
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

        # Regra avançada: Não pode remover peça que já está em trilha
        # (A menos que TODAS as peças do inimigo estejam em trilha)
        if self.is_part_of_mill(pos_idx, opponent):
            if not self.all_enemies_in_mill(opponent):
                return False

        return True

    # --- AÇÕES (Executores) ---

    def apply_place(self, pos_idx):
        """Executa a colocação e atualiza estado."""
        if not self.is_valid_place(pos_idx):
            raise ValueError(f"Movimento inválido: Place {pos_idx}")

        self.board[pos_idx] = self.turn
        self.pieces_to_place[self.turn] -= 1
        self.pieces_on_board[self.turn] += 1

        # Checa se formou trilha
        if self.check_mill_formed(pos_idx):
            self.pending_removal = True
        else:
            self._switch_turn_logic()

    def apply_move(self, start_pos, end_pos):
        """Executa o movimento."""
        if not self.is_valid_move(start_pos, end_pos):
            raise ValueError(f"Movimento inválido: Move {start_pos}->{end_pos}")

        self.board[start_pos] = None
        self.board[end_pos] = self.turn

        if self.check_mill_formed(end_pos):
            self.pending_removal = True
        else:
            self._switch_turn_logic()

    def apply_remove(self, pos_idx):
        """Executa a remoção."""
        if not self.is_valid_remove(pos_idx):
            raise ValueError("Remoção inválida")

        opponent = "R" if self.turn == "V" else "V"
        self.board[pos_idx] = None
        self.pieces_on_board[opponent] -= 1
        self.pending_removal = False

        self._switch_turn_logic()

    # --- AUXILIARES INTERNOS ---

    

    def check_mill_formed(self, pos_idx):
        """Verifica se a peça recém colocada/movida em pos_idx fechou uma trilha."""
        player = self.board[pos_idx]
        for mill in self.POSSIBLE_MILLS:
            if pos_idx in mill:
                # Verifica se as outras 2 posições são do mesmo jogador
                if all(self.board[p] == player for p in mill):
                    return True
        return False
    
    def is_part_of_mill(self, pos_idx, player):
        """Verifica se a peça na posição pos_idx faz parte de uma trilha formada."""
        # Se a peça nem é do jogador, não é trilha dele
        if self.board[pos_idx] != player:
            return False
            
        for mill in self.POSSIBLE_MILLS:
            if pos_idx in mill:
                # Verifica se as outras 2 casas da trilha também são do jogador
                if all(self.board[p] == player for p in mill):
                    return True
        return False

    def all_enemies_in_mill(self, player):
        """Verifica se TODAS as peças do 'player' estão protegidas em trilhas."""
        # Lista todas as posições onde o inimigo tem peças
        indices_player = [i for i, p in enumerate(self.board) if p == player]
        
        if not indices_player:
            return False # Se não tem peças, irrelevante
            
        # Se acharmos UMA peça que não está em trilha, retorna False (pode remover essa)
        for idx in indices_player:
            if not self.is_part_of_mill(idx, player):
                return False
                
        # Se chegou aqui, é porque todas estão em trilha (então a regra de proteção cai)
        return True

    def _switch_turn_logic(self):
        """Troca o turno e verifica transição de fase."""
        # Troca jogador
        self.turn = "R" if self.turn == "V" else "V"

        # Se acabaram as peças de colocar de AMBOS, muda fase
        if self.pieces_to_place["V"] == 0 and self.pieces_to_place["R"] == 0:
            self.phase = "MOVEMENT"

    def check_winner(self):
        """Retorna 'V', 'R', 'Draw' ou None (jogo segue)."""
        # Só checa vitória na fase de movimento
        if self.phase == "MOVEMENT":
            if self.pieces_on_board["V"] < 3:
                return "R"
            if self.pieces_on_board["R"] < 3:
                return "V"

            # TODO: Checar se o jogador atual está "trancado" (sem movimentos válidos)
            # Isso é condição de derrota também.

        return None
