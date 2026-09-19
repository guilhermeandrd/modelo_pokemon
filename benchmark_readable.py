# benchmark_readable.py
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

# mapeamento oficial de enums da libcg para validação humana
OPTION_TYPE_MAP = {
    0: "NUMBER", 1: "YES", 2: "NO", 3: "SELECIONAR_CARTA", 4: "TOOL_CARD", 
    5: "ENERGY_CARD", 6: "ENERGY", 7: "JOGAR_CARTA_MAO", 8: "LIGAR_ENERGIA", 
    9: "EVOLUIR_POKEMON", 10: "USAR_HABILIDADE", 11: "DISCARD", 12: "RECUAR", 
    13: "ATACAR", 14: "PASSAR_TURNO", 15: "ORDEM_SKILL", 16: "SPECIAL_CONDITION"
}

def read_deck_csv(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as file:
        csv_lines = file.read().strip().split("\n")
    return [int(csv_lines[i].strip()) for i in range(min(60, len(csv_lines)))]

def parse_option_to_text(opt_dict: dict) -> str:
    """Converte o objeto de opção da libcg em uma descrição textual clara."""
    if not isinstance(opt_dict, dict):
        return "PASSAR_TURNO"

    t_val = opt_dict.get("type", -1)
    label = OPTION_TYPE_MAP.get(t_val, f"AÇÃO_{t_val}")
    card_id = opt_dict.get("cardId")
    attack_id = opt_dict.get("attackId")

    if t_val == 13: # ATTACK
        return f"ATACAR (Ataque ID: {attack_id if attack_id is not None else 'Ativo'})"
    elif t_val == 8: # ATTACH
        return f"LIGAR_ENERGIA (Carta ID: {card_id if card_id is not None else 'Mão'})"
    elif t_val == 7: # PLAY
        return f"JOGAR_CARTA_MAO (Carta ID: {card_id if card_id is not None else 'Treinador/Pokémon'})"
    elif t_val == 9: # EVOLVE
        return f"EVOLUIR (Carta ID: {card_id if card_id is not None else 'Campo'})"
    elif t_val == 10: # ABILITY
        return f"USAR_HABILIDADE (Carta ID: {card_id if card_id is not None else 'Ativa'})"
    elif t_val == 12: # RETREAT
        return "RECUAR"
    elif t_val == 14: # END
        return "PASSAR_TURNO"
    else:
        card_str = f" [Carta ID: {card_id}]" if card_id is not None else ""
        return f"{label}{card_str}"

def run_human_validation_benchmark(num_matches=5):
    print(f"🚀 Gerando relatório legível de {num_matches} partidas para Validação Humana...")
    deck = read_deck_csv(DECK_PATH)
    detailed_logs = []

    for idx in range(num_matches):
        res = battle_start(deck, deck)
        obs_dict = res[0] if isinstance(res, tuple) else res
        turn = 0

        while obs_dict and turn < 150:
            turn += 1
            curr_state = obs_dict.get("current", {}) if isinstance(obs_dict, dict) else {}
            p_idx = curr_state.get("yourIndex", 0) if curr_state else 0
            
            agent_func = rule_agent if p_idx == 0 else random_agent
            agent_name = "Rule-Based (P1)" if p_idx == 0 else "Random (P2)"
            
            # escolha do agente
            action_indices = agent_func(obs_dict)
            
            # converte os índices escolhidos para os textos reais correspondentes
            select_data = obs_dict.get("select", {}) if isinstance(obs_dict, dict) else {}
            options = select_data.get("option", []) if isinstance(select_data, dict) else []
            
            action_texts = []
            for a_idx in action_indices:
                if 0 <= a_idx < len(options):
                    action_texts.append(parse_option_to_text(options[a_idx]))
                else:
                    action_texts.append("PASSAR_TURNO")

            action_text_str = " + ".join(action_texts) if action_texts else "PASSAR_TURNO"

            # grava no log estruturado
            detailed_logs.append({
                "Partida": idx + 1,
                "Turno": turn,
                "Agente": agent_name,
                "Ação Executada": action_text_str,
                "Índice Escolhido": str(action_indices)
            })

            try:
                obs_dict = battle_select(action_indices)
                if isinstance(obs_dict, dict) and obs_dict.get("current", {}).get("result", -1) != -1:
                    break
            except Exception:
                break

        battle_finish()

    df = pd.DataFrame(detailed_logs)
    df.to_csv("validacao_humana_acoes.csv", index=False)
    
    print("\n✅ Relatório gerado com sucesso!")
    print(df.head(15).to_string(index=False))
    print("\n📄 Arquivo salvo como: 'validacao_humana_acoes.csv'")

if __name__ == "__main__":
    run_human_validation_benchmark(5)