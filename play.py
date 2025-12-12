import time
import torch
import numpy as np
import os
import sys

# Ajuste de path para importar módulos
sys.path.append(os.getcwd())

from envs.game_logic import TrilhaGame
from envs.trilha_gym import TrilhaEnv
from train import RandomAgent, TrainedModelAgent, DEVICE

# --- CONFIGURAÇÃO VISUAL ---
COLOR_V = "🔴"  # Jogador V (Vermelho)
COLOR_R = "🔵"  # Jogador R (Azul)
COLOR_EMPTY = "⚫"  # Vazio (ou use "⚪" ou "  ")
COLOR_LINE = "➖"
COLOR_PIPE = "│"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print("=" * 40)
    print("   TRILHA RL - ARENA DE BATALHA   ")
    print("=" * 40)


def render_board(game):
    """
    Renderiza o tabuleiro de forma bonita no terminal usando a matriz 7x7.
    O jogo usa índices lineares (0-23), então precisamos mapear.
    """
    # Mapeamento visual reverso (Coord visual -> Índice Lógico)
    # Apenas para desenhar. Se for None, é espaço vazio não jogável.

    # Criamos uma matriz 7x7 visual
    grid = [[None for _ in range(7)] for _ in range(7)]

    # Mapeamento do envs/trilha_gym.py (idx_to_coord)
    idx_map = {
        0: (0, 0),
        1: (0, 3),
        2: (0, 6),
        3: (1, 1),
        4: (1, 3),
        5: (1, 5),  # Corrigindo ordem visual concêntrica
        # Ops, a lógica do jogo é Anel Externo -> Médio -> Interno.
        # Vamos usar o board state direto.
    }

    # Vamos usar o board direto e colocar nas posições manuais para ficar bonito
    b = game.board

    def p(idx):
        piece = b[idx]
        if piece == "V":
            return COLOR_V
        if piece == "R":
            return COLOR_R
        return "⚪"  # Ponto vazio jogável

    # Layout ASCII Hardcoded para Trilha
    # Indices baseados no game_logic.py:
    # Ext: 0..7 | Med: 8..15 | Int: 16..23

    print(f"\n   {p(0)}──────────────{p(1)}──────────────{p(2)}")
    print("   │              │              │")
    print(f"   │      {p(8)}───────{p(9)}───────{p(10)}      │")
    print("   │      │       │       │      │")
    print(f"   │      │   {p(16)}───{p(17)}───{p(18)}   │      │")
    print("   │      │   │       │   │      │")
    print(f"   {p(7)}──────{p(15)}───{p(23)}       {p(19)}───{p(11)}──────{p(3)}")
    print("   │      │   │       │   │      │")
    print(f"   │      │   {p(22)}───{p(21)}───{p(20)}   │      │")
    print("   │      │       │       │      │")
    print(f"   │      {p(14)}───────{p(13)}───────{p(12)}      │")
    print("   │              │              │")
    print(f"   {p(6)}──────────────{p(5)}──────────────{p(4)}\n")

    # Placar
    print(f"   Turno: {COLOR_V if game.turn == 'V' else COLOR_R}")
    print(f"   Fase: {game.phase}")
    if game.pending_removal:
        print("   ⚠️  TRILHA FORMADA! REMOVENDO PEÇA... ⚠️")
    print(
        f"   Peças {COLOR_V}: {game.pieces_on_board['V']} (Mão: {game.pieces_to_place['V']})"
    )
    print(
        f"   Peças {COLOR_R}: {game.pieces_on_board['R']} (Mão: {game.pieces_to_place['R']})"
    )
    print("-" * 40)


def get_agent_choice(player_name):
    print(f"\nEscolha o Agente para o Jogador {player_name}:")
    print("1. 🎲 Aleatório (Random)")
    print("2. 🧠 Modelo Fase 1 (Vs Random)")
    print("3. 🏆 Modelo Fase 2 (Vs Expert)")

    while True:
        try:
            choice = input("Opção (1-3): ")
            if choice == "1":
                return "RANDOM", None
            if choice == "2":
                return "MODEL", "model_vs_random.pth"
            if choice == "3":
                return "MODEL", "model_vs_expert.pth"
        except KeyboardInterrupt:
            sys.exit()


def create_agent(type, path, env):
    if type == "RANDOM":
        return RandomAgent()
    elif type == "MODEL":
        if not os.path.exists(path):
            print(f"❌ Erro: Modelo '{path}' não encontrado! Treine primeiro.")
            sys.exit()
        print(f"Carregando {path}...")
        return TrainedModelAgent(path, env)


def run_match(agent_v, agent_r, delay=0.5):
    env = TrilhaEnv()
    obs, info = env.reset()
    game = env.game

    # Mapear agentes
    agents = {"V": agent_v, "R": agent_r}

    done = False

    while not done:
        clear_screen()
        print_header()
        render_board(game)

        current_player = game.turn
        current_agent = agents[current_player]

        # Pega ação
        # Nota: TrainedModelAgent precisa do mask e state
        mask = env.get_action_mask()

        # Pequeno delay para visualização
        time.sleep(delay)

        print(f"🤔 {current_player} está pensando...")

        # O TrainedModelAgent foi feito para jogar como "V" (Player 1 da visão da rede).
        # Se ele estiver jogando como "R", precisamos inverter a observação?
        # A classe TrilhaEnv atual já inverte a observação no _get_obs() baseada no turno!
        # Então o obs[0] é sempre "minhas peças" e obs[1] "inimigo".
        # O modelo pode jogar de qualquer lado sem mexer nos dados.

        action = current_agent.act(obs, mask, game)

        # Executa no ambiente
        # Importante: Como estamos rodando manualmente o loop, usamos game.apply direto?
        # Não, usamos env.step para manter compatibilidade com a lógica de recompensa/regra
        # Mas o env.step atual executa DOIS turnos (Agente e Oponente).
        # Precisamos de um step "unitário" para visualização passo a passo.
        # Vamos interagir direto com a lógica do jogo (game) ou adaptar o env.

        # Para visualização, é melhor chamar a lógica do jogo diretamente com proteções,
        # pois o env.step foi desenhado para treino (turnos acoplados).

        try:
            if action < 24:
                game.apply_place(action)
            else:
                # Decodificar movimento
                move_idx = action - 24
                start = move_idx // 4
                direction = move_idx % 4
                dirs = ["d", "e", "c", "b"]
                target = game.ADJACENCY[start][dirs[direction]]
                game.apply_move(start, target)

            # Auto-remoção visual
            if game.pending_removal:
                clear_screen()
                print_header()
                render_board(game)
                print(f"⚔️  {current_player} fez trilha! Removendo peça...")
                time.sleep(delay)

                # Tenta remover (Lógica simplificada igual ao treino: primeira válida)
                removed = False
                opp = "R" if current_player == "V" else "V"
                for i, p in enumerate(game.board):
                    if p == opp:
                        try:
                            game.apply_remove(i)
                            removed = True
                            break
                        except:
                            continue

                if not removed:
                    # Caso raro onde não dá pra remover nada (não deveria acontecer mais)
                    game.pending_removal = False
                    game._switch_turn_logic()

        except Exception as e:
            print(f"❌ Erro Crítico: {e}")
            break

        # Verifica vitória
        winner = game.check_winner()
        if winner:
            clear_screen()
            print_header()
            render_board(game)
            print(
                f"\n🎉🎉 VITORIA DO JOGADOR {COLOR_V if winner == 'V' else COLOR_R} ({winner})! 🎉🎉"
            )
            break

        # Atualiza obs para o próximo
        obs = env._get_obs()


def main():
    clear_screen()
    print_header()

    # Setup Env dummy para carregar modelos (pegar shapes)
    dummy_env = TrilhaEnv()

    print("Configuração da Partida:")
    type_v, path_v = get_agent_choice(f"{COLOR_V} (Vermelho/Primeiro)")
    type_r, path_r = get_agent_choice(f"{COLOR_R} (Azul/Segundo)")

    agent_v = create_agent(type_v, path_v, dummy_env)
    agent_r = create_agent(type_r, path_r, dummy_env)

    print("\nIniciando partida em 3 segundos...")
    time.sleep(3)

    run_match(agent_v, agent_r, delay=0.5)


if __name__ == "__main__":
    main()
