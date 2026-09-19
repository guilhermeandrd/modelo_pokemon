# benchmark_verbose.py
import os
import sys
import pandas as pd

SAMPLE_DIR = os.environ.get("CABT_SAMPLE_SUBMISSION_DIR", "sample_submission")
if os.path.abspath(SAMPLE_DIR) not in sys.path:
    sys.path.append(os.path.abspath(SAMPLE_DIR))

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class
from rule_agent import agent as rule_agent
from random_agent import agent as random_agent

DECK_PATH = os.path.join(SAMPLE_DIR, "deck.csv")

def read_deck_csv(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as file:
        csv_lines = file.read().strip().split("\n")
    return [int(csv_lines[i].strip()) for i in range(min(60, len(csv_lines)))]

def run_detailed_benchmark(num_matches=10):
    deck = read_deck_csv(DECK_PATH)
    action_logs = []

    for idx in range(num_matches):
        obs_dict, _ = battle_start(deck, deck)
        turn = 0

        while obs_dict and turn < 150:
            turn += 1
            curr_state = obs_dict.get("current", {})
            p_idx = curr_state.get("yourIndex", 0) if curr_state else 0
            
            agent_func = rule_agent if p_idx == 0 else random_agent
            agent_name = "Rule-Based" if p_idx == 0 else "Random"
            
            action = agent_func(obs_dict)
            
            # Extrai o nome legível da ação
            obs = to_observation_class(obs_dict)
            action_desc = "PASS"
            if obs and obs.select and obs.select.option and action[0] < len(obs.select.option):
                opt_type = obs.select.option[action[0]].type
                action_desc = opt_type.name if hasattr(opt_type, 'name') else str(opt_type)

            action_logs.append({
                "match_id": idx + 1,
                "turn": turn,
                "player": agent_name,
                "chosen_action_index": action[0],
                "action_type": action_desc
            })

            try:
                obs_dict = battle_select(action)
                if obs_dict and obs_dict.get("current", {}).get("result", -1) != -1:
                    break
            except Exception:
                break

        battle_finish()

    pd.DataFrame(action_logs).to_csv("detailed_action_logs.csv", index=False)
    print("✅ Logs detalhados exportados para 'detailed_action_logs.csv'!")

if __name__ == "__main__":
    run_detailed_benchmark(10)