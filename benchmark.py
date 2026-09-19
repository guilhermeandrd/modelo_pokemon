# benchmark.py
import os
import sys
import pandas as pd

SAMPLE_DIR = os.environ.get("CABT_SAMPLE_SUBMISSION_DIR", "sample_submission")
if os.path.abspath(SAMPLE_DIR) not in sys.path:
    sys.path.append(os.path.abspath(SAMPLE_DIR))

from cg.game import battle_start, battle_select, battle_finish
from rule_agent import agent as rule_agent
from random_agent import agent as random_agent

DECK_PATH = os.path.join(SAMPLE_DIR, "deck.csv")

def read_deck_csv(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as file:
        csv_lines = file.read().strip().split("\n")
    return [int(csv_lines[i].strip()) for i in range(min(60, len(csv_lines)))]

def run_benchmark(num_matches=50):
    print(f"🚀 Executando Benchmark NATIVO de {num_matches} partidas...")
    deck = read_deck_csv(DECK_PATH)
    results = []

    for idx in range(num_matches):
        # 1. onicia a batalha na C++ API (retorna tupla: (obs_dict, start_data))
        obs_dict, start_data = battle_start(deck, deck)
        
        turn_count = 0
        max_turns = 200
        winner = -1

        while obs_dict and turn_count < max_turns:
            turn_count += 1
            
            # pega o jogador que deve tomar a acao no turno atual
            current_state = obs_dict.get("current", {})
            current_player = current_state.get("yourIndex", 0) if current_state else 0

            # seleciona o agente correto com base no yourIndex da observação
            if current_player == 0:
                action = rule_agent(obs_dict)
            else:
                action = random_agent(obs_dict)

            # executa o passo e recebe o dicionario do novo estado
            try:
                obs_dict = battle_select(action)
            except IndexError:
                # caso a libcg sinalize fim de opcoes legais
                break

            # verifica condicao de termino
            if obs_dict and obs_dict.get("current", {}).get("result", -1) != -1:
                winner = obs_dict["current"]["result"]
                break

        # finaliza a memoria da c++ lib
        battle_finish()

        w_str = "Rule-Based" if winner == 0 else ("Random" if winner == 1 else "Empate/Timeout")
        results.append({
            "match_id": idx + 1,
            "winner": w_str,
            "winner_id": winner,
            "total_turns": turn_count
        })
        print(f"Partida {idx + 1:02d}/{num_matches} | Vencedor: {w_str:<12} | Turnos: {turn_count}")

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    
    print("\n📊 Resumo Estatístico para o Artigo:")
    print(df["winner"].value_counts(normalize=True) * 100)
    print("\nResultados salvos em 'benchmark_results.csv'")

if __name__ == "__main__":
    run_benchmark(50)