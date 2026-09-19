# server.py
import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

SAMPLE_DIR = os.environ.get("CABT_SAMPLE_SUBMISSION_DIR", "sample_submission")
if os.path.abspath(SAMPLE_DIR) not in sys.path:
    sys.path.append(os.path.abspath(SAMPLE_DIR))

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, OptionType
from rule_agent import agent as rule_agent
from random_agent import agent as random_agent

app = FastAPI()

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

game_session = {
    "obs_dict": None,
    "logs": [],
    "turn_count": 0,
    "active": False
}

DECK_PATH = os.path.join(SAMPLE_DIR, "deck.csv")

# mapeamento numerico dos OptionType do Kaggle para descricoes em portugues
OPTION_TYPE_MAP = {
    0: "NUMBER",
    1: "YES",
    2: "NO",
    3: "SELECIONAR CARTA",
    4: "TOOL_CARD",
    5: "ENERGY_CARD",
    6: "ENERGY",
    7: "JOGAR CARTA DA MÃO",
    8: "LIGAR ENERGIA",
    9: "EVOLUIR POKÉMON",
    10: "HABILIDADE",
    11: "DESGARTAR",
    12: "RECUAR POKÉMON",
    13: "ATACAR",
    14: "PASSAR TURNO",
    15: "ORDEM DE SKILL",
    16: "CONDIÇÃO ESPECIAL"
}

def read_deck_csv(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as file:
        csv_lines = file.read().strip().split("\n")
    return [int(csv_lines[i].strip()) for i in range(min(60, len(csv_lines)))]

def describe_option_dict(opt):
    """traduz o dicionario de opção vindo do simulador para texto explicativo."""
    if not isinstance(opt, dict):
        # se for dataclass
        opt_type_val = getattr(opt, 'type', None)
        type_int = int(opt_type_val) if opt_type_val is not None else -1
        card_id = getattr(opt, 'cardId', None)
        attack_id = getattr(opt, 'attackId', None)
    else:
        type_int = opt.get("type", -1)
        card_id = opt.get("cardId")
        attack_id = opt.get("attackId")

    label = OPTION_TYPE_MAP.get(type_int, f"AÇÃO ({type_int})")

    if type_int == 13: # ATTACK
        return f"⚔️ ATACOU com o Pokémon Ativo (Ataque ID: {attack_id if attack_id is not None else 'Principal'})"
    elif type_int == 8: # ATTACH
        return f"⚡ LIGOU ENERGIA no Pokémon (Carta: {card_id if card_id is not None else 'Mão'})"
    elif type_int == 7: # PLAY
        return f"🃏 JOGOU CARTA DA MÃO (Carta ID: {card_id if card_id is not None else 'Treinador/Pokémon'})"
    elif type_int == 9: # EVOLVE
        return f"🧬 EVOLUIU Pokémon (Carta ID: {card_id if card_id is not None else 'Campo'})"
    elif type_int == 10: # ABILITY
        return f"✨ USOU HABILIDADE (Carta ID: {card_id if card_id is not None else 'Ativa'})"
    elif type_int == 12: # RETREAT
        return "🔄 RECUOU Pokémon Ativo para o Banco"
    elif type_int == 14: # END
        return "PASSOU O TURNO"
    else:
        card_str = f" [Carta ID: {card_id}]" if card_id is not None else ""
        return f"{label}{card_str}"

def extract_legal_actions(obs_dict):
    """Extrai a lista de opções legais diretamente do dicionario de observação."""
    if not obs_dict or not isinstance(obs_dict, dict):
        return []
    
    select_data = obs_dict.get("select")
    if not select_data or not isinstance(select_data, dict):
        return ["Aguardando seleção / Fim de Turno"]

    options = select_data.get("option", [])
    if not options:
        return ["Sem opções legais no momento"]

    actions = []
    for idx, opt in enumerate(options):
        actions.append(f"[{idx}] {describe_option_dict(opt)}")
    return actions

def get_action_log(obs_dict, action_indices, agent_name, turn):
    """Gera a mensagem do log para as ações escolhidas."""
    if not obs_dict or not isinstance(obs_dict, dict):
        return f"Turno {turn} | {agent_name}: Executou escolha {action_indices}"

    select_data = obs_dict.get("select")
    if not select_data or not isinstance(select_data, dict):
        return f"Turno {turn} | {agent_name}: Concluiu o turno"

    options = select_data.get("option", [])
    executed_descs = []
    
    for idx in action_indices:
        if 0 <= idx < len(options):
            executed_descs.append(describe_option_dict(options[idx]))

    if executed_descs:
        return f"Turno {turn} | {agent_name}: " + " + ".join(executed_descs)
    return f"Turno {turn} | {agent_name}: Executou índice {action_indices}"

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.post("/api/start")
def start_game():
    deck = read_deck_csv(DECK_PATH)
    res = battle_start(deck, deck)
    obs_dict = res[0] if isinstance(res, tuple) else res
    
    game_session["obs_dict"] = obs_dict
    game_session["logs"] = ["Partida iniciada no simulador TCG!"]
    game_session["turn_count"] = 0
    game_session["active"] = True
    
    return get_current_state()

@app.get("/api/state")
def get_current_state():
    obs_dict = game_session["obs_dict"]
    if not game_session["active"] or not obs_dict:
        return {"active": False}

    current_state = obs_dict.get("current", {}) if isinstance(obs_dict, dict) else {}
    current = current_state.get("yourIndex", 0) if current_state else 0
    result = current_state.get("result", -1) if current_state else -1

    return {
        "active": True,
        "is_terminal": (result != -1),
        "winner": result,
        "current_player": current,
        "current_player_name": "Rule-Based Agent (P1)" if current == 0 else "Random Agent (P2)",
        "turn": game_session["turn_count"],
        "legal_actions": extract_legal_actions(obs_dict),
        "logs": game_session["logs"]
    }

@app.post("/api/step")
def step_turn():
    obs_dict = game_session["obs_dict"]
    if not game_session["active"] or not obs_dict:
        return get_current_state()

    current_state = obs_dict.get("current", {}) if isinstance(obs_dict, dict) else {}
    current = current_state.get("yourIndex", 0) if current_state else 0
    
    if current == 0:
        action = rule_agent(obs_dict)
        agent_name = "Rule-Based (P1)"
    else:
        action = random_agent(obs_dict)
        agent_name = "Random (P2)"

    # registra o log traduzido antes de avançar a libcg
    log_line = get_action_log(obs_dict, action, agent_name, game_session["turn_count"] + 1)
    game_session["logs"].append(log_line)

    try:
        new_obs_dict = battle_select(action)
        game_session["obs_dict"] = new_obs_dict
        game_session["turn_count"] += 1

        res = new_obs_dict.get("current", {}).get("result", -1) if isinstance(new_obs_dict, dict) else -1
        if res != -1:
            w_name = "Rule-Based (P1)" if res == 0 else ("Random (P2)" if res == 1 else "Empate")
            game_session["logs"].append(f"🏆 Partida Finalizada! Vencedor: {w_name}")
            battle_finish()
    except IndexError:
        game_session["logs"].append("Fim de opções legais no simulador.")
        game_session["active"] = False

    return get_current_state()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)