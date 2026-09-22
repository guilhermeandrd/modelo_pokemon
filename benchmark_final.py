import os
import sys
import pandas as pd
from collections import defaultdict

SAMPLE_DIR = os.environ.get("CABT_SAMPLE_SUBMISSION_DIR", "sample_submission")
if os.path.abspath(SAMPLE_DIR) not in sys.path:
    sys.path.append(os.path.abspath(SAMPLE_DIR))

from cg.game import battle_start, battle_select, battle_finish
from rule_agent import agent as rule_agent
from random_agent import agent as random_agent
from q_learning import agent as QlearningAgent

DECK_PATH = os.path.join(SAMPLE_DIR, "deck.csv")

def read_deck(path):
    with open(path, "r") as f:
        return [int(x.strip()) for x in f.read().strip().splitlines()[:60]]

def run_matchup(agent1_func, name1, agent2_func, name2, num_games=100):
    deck = read_deck(DECK_PATH)
    wins = {name1: 0, name2: 0, "Empates": 0}
    turns_history = []

    print(f"  ⚔️ Confronto: {name1} VS {name2} ({num_games} partidas)...")

    for g in range(num_games):
        if g % 2 == 0:
            p0_func, p0_name = agent1_func, name1
            p1_func, p1_name = agent2_func, name2
        else:
            p0_func, p0_name = agent2_func, name2
            p1_func, p1_name = agent1_func, name1

        res = battle_start(deck, deck)
        obs = res[0] if isinstance(res, tuple) else res
        turns = 0

        while obs and turns < 150:
            turns += 1
            curr = obs.get("current", {}) if isinstance(obs, dict) else getattr(obs, "current", {})
            p_idx = curr.get("yourIndex", 0) if isinstance(curr, dict) else getattr(curr, "yourIndex", 0)

            active_agent = p0_func if p_idx == 0 else p1_func
            action = active_agent(obs)

            try:
                obs = battle_select(action)
                result = obs.get("current", {}).get("result", -1) if isinstance(obs, dict) else -1
                if result != -1:
                    winner_name = p0_name if result == 0 else p1_name
                    wins[winner_name] += 1
                    turns_history.append(turns)
                    break
            except Exception:
                wins["Empates"] += 1
                break

        battle_finish()

    avg_turns = sum(turns_history) / len(turns_history) if turns_history else 0
    return wins, avg_turns

def run_battery(bateria_num: int, matches_per_pair=100):
    print(f"\n🚀 === INICIANDO BATERIA {bateria_num} ({matches_per_pair} partidas por confronto) ===")
    
    results = []

    w1, t1 = run_matchup(rule_agent, "Rule-Based", random_agent, "Random", matches_per_pair)
    results.append({
        "Bateria": bateria_num,
        "Confronto": "Rule-Based vs Random", 
        "Vitórias P1": f"Rule-Based ({w1['Rule-Based']})", 
        "Vitórias P2": f"Random ({w1['Random']})", 
        "Média Turnos": round(t1, 1)
    })

    w2, t2 = run_matchup(QlearningAgent, "Q-learning", random_agent, "Random", matches_per_pair)
    results.append({
        "Bateria": bateria_num,
        "Confronto": "Q-learning vs Random", 
        "Vitórias P1": f"Q-learning ({w2['Q-learning']})", 
        "Vitórias P2": f"Random ({w2['Random']})", 
        "Média Turnos": round(t2, 1)
    })

    w3, t3 = run_matchup(rule_agent, "Rule-Based", QlearningAgent, "Q-learning", matches_per_pair)
    results.append({
        "Bateria": bateria_num,
        "Confronto": "Rule-Based vs Q-learning", 
        "Vitórias P1": f"Rule-Based ({w3['Rule-Based']})", 
        "Vitórias P2": f"Q-learning ({w3['Q-learning']})", 
        "Média Turnos": round(t3, 1)
    })

    df_res = pd.DataFrame(results)
    
    file_name = f"bateria_{bateria_num}.csv"
    df_res.to_csv(file_name, index=False)

    print(f"✅ Bateria {bateria_num} finalizada e salva em: {file_name}")
    print(df_res.to_string(index=False))
    
    return df_res

def main():
    MATCHES_PER_PAIR = 100
    NUM_BATTERIES = 5  
    
    all_batteries = []

    for b in range(1, NUM_BATTERIES + 1):
        df_b = run_battery(bateria_num=b, matches_per_pair=MATCHES_PER_PAIR)
        all_batteries.append(df_b)

    df_consolidado = pd.concat(all_batteries, ignore_index=True)
    df_consolidado.to_csv("resultado_consolidado_baterias.csv", index=False)
    
    print("\n🏆 === EXECUÇÃO CONCLUÍDA ===")
    print("Todas as baterias foram salvas separadamente (bateria_1.csv a bateria_5.csv).")

if __name__ == "__main__":
    main()