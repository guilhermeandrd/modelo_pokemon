# debug_benchmark.py
import os
import sys

SAMPLE_DIR = os.environ.get("CABT_SAMPLE_SUBMISSION_DIR", "sample_submission")
if os.path.abspath(SAMPLE_DIR) not in sys.path:
    sys.path.append(os.path.abspath(SAMPLE_DIR))

try:
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class
    print("✅ Módulo 'cg' importado com sucesso!")
except ImportError as e:
    print(f"❌ Erro ao importar 'cg': {e}")
    sys.exit(1)

DECK_PATH = os.path.join(SAMPLE_DIR, "deck.csv") if os.path.exists(os.path.join(SAMPLE_DIR, "deck.csv")) else "deck.csv"

def validate_deck(path):
    if not os.path.exists(path):
        print(f"❌ Arquivo de deck não encontrado em '{path}'. Criando deck genérico de teste...")
        return [1] * 60  
    
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip().isdigit()]
    
    print(f"📄 Deck lido com {len(lines)} cartas.")
    if len(lines) < 60:
        print("⚠️ ALERTA: Deck possui menos de 60 cartas! Preenchendo até 60...")
        lines += [lines[0]] * (60 - len(lines))
    return [int(x) for x in lines[:60]]

def test_single_match(agent_func, agent_name):
    print(f"\n--- Iniciando Partida de Teste com: {agent_name} vs Random ---")
    deck = validate_deck(DECK_PATH)
    
    from random_agent import agent as random_agent

    res = battle_start(deck, deck)
    obs_dict = res[0] if isinstance(res, tuple) else res
    turn = 0

    while obs_dict and turn < 50:
        turn += 1
        curr = obs_dict.get("current", {}) if isinstance(obs_dict, dict) else getattr(obs_dict, "current", {})
        p_idx = curr.get("yourIndex", 0) if isinstance(curr, dict) else getattr(curr, "yourIndex", 0)

        active_agent = agent_func if p_idx == 0 else random_agent
        
        try:
            action = active_agent(obs_dict)
            
            select_data = obs_dict.get("select", {}) if isinstance(obs_dict, dict) else getattr(obs_dict, "select", {})
            min_c = select_data.get("minCount", 1) if isinstance(select_data, dict) else getattr(select_data, "minCount", 1)
            options = select_data.get("option", []) if isinstance(select_data, dict) else getattr(select_data, "option", [])
            
            if len(action) < min_c and len(options) >= min_c:
                print(f"⚠️ AVISO no Turno {turn}: {agent_name} retornou {len(action)} ações, mas o mínimo exigido era {min_c}.")

            obs_dict = battle_select(action)
            res_val = obs_dict.get("current", {}).get("result", -1) if isinstance(obs_dict, dict) else -1
            
            if res_val != -1:
                winner = "P1" if res_val == 0 else "P2"
                print(f"🏆 Partida encerrada no Turno {turn}! Vencedor: {winner} ({'Vitória' if res_val == 0 else 'Derrota'}).")
                break

        except Exception as e:
            print(f"💥 ERRO CRÍTICO no Turno {turn} com {agent_name}: {e}")
            import traceback
            traceback.print_exc()
            break

    battle_finish()

if __name__ == "__main__":
    from stochastic_agent import agent as stochastic_agent
    test_single_match(stochastic_agent, "Stochastic Agent")