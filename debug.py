# Arquivo: debug.py
import time
import numpy as np
from envs.trilha_gym import TrilhaEnv

def main():
    print(">>> Testando a Física do Jogo...")
    try:
        env = TrilhaEnv()
        env.reset()
        print(">>> Ambiente carregado com sucesso!")
    except Exception as e:
        print(f"ERRO ao criar ambiente: {e}")
        return

    # Vamos tentar fazer movimentos aleatórios válidos
    print("\n>>> Iniciando Random Walk (Testando Mascaramento)...")
    for i in range(5):
        mask = env.get_action_mask()
        valid_actions = np.where(mask == 1)[0]
        
        if len(valid_actions) == 0:
            print("ERRO: Nenhuma ação válida (Jogo travou?)")
            break
            
        action = np.random.choice(valid_actions)
        print(f"Passo {i+1}: Ação {action} selecionada de {len(valid_actions)} opções válidas.")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print("Jogo terminou cedo!")
            break
            
    print("\n>>> SUCESSO: O grafo e o mascaramento parecem estáveis.")

if __name__ == "__main__":
    main()