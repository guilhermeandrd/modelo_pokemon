# benchmark_narrative.py
import os
import sys
import pandas as pd

SAMPLE_DIR = os.environ.get("CABT_SAMPLE_SUBMISSION_DIR", "sample_submission")
if os.path.abspath(SAMPLE_DIR) not in sys.path:
    sys.path.append(os.path.abspath(SAMPLE_DIR))

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, all_card_data, all_attack
from rule_agent import agent as rule_agent
from random_agent import agent as random_agent

DECK_PATH = os.path.join(SAMPLE_DIR, "deck.csv")
CARD_DATA_PATH = "EN Card Data.csv"

# mapeamento de Cartas a partir do EN Card Data.csv
CARD_INFO_MAP = {}
if os.path.exists(CARD_DATA_PATH):
    try:
        df_cards = pd.read_csv(CARD_DATA_PATH)
        for _, row in df_cards.iterrows():
            c_id = row.get("Card ID")
            if pd.notna(c_id):
                c_id = int(c_id)
                if c_id not in CARD_INFO_MAP:
                    CARD_INFO_MAP[c_id] = {
                        "name": str(row.get("Card Name", "")),
                        "type": str(row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")),
                        "hp": int(row["HP"]) if pd.notna(row.get("HP")) else None
                    }
    except Exception as e:
        print(f"Aviso ao ler {CARD_DATA_PATH}: {e}")

try:
    for c in all_card_data():
        if c.cardId not in CARD_INFO_MAP:
            CARD_INFO_MAP[c.cardId] = {"name": c.name, "type": "Pokémon", "hp": c.hp}
except Exception:
    pass

ATTACK_NAME_MAP = {}
try:
    ATTACK_NAME_MAP = {a.attackId: a.name for a in all_attack()}
except Exception:
    pass

OPTION_TYPE_NAMES = {
    0: "NUMBER", 1: "YES", 2: "NO", 3: "SELECIONAR_CARTA", 4: "TOOL_CARD", 
    5: "ENERGY_CARD", 6: "ENERGY", 7: "JOGAR_CARTA", 8: "LIGAR_ENERGIA", 
    9: "EVOLUIR", 10: "USAR_HABILIDADE", 11: "DISCARD", 12: "RECUAR", 
    13: "ATACAR", 14: "PASSAR_TURNO", 15: "ORDEM_SKILL", 16: "SPECIAL_CONDITION"
}

def read_deck_csv(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as file:
        content = file.read().strip()
        csv_lines = content.splitlines()
    return [int(csv_lines[i].strip()) for i in range(min(60, len(csv_lines)))]

def get_card_info(card_id, fallback="Carta Oculta"):
    if card_id is None or card_id == "" or card_id == -1 or card_id == 0:
        return fallback, "Oculta", None
    try:
        c_id = int(card_id)
        if c_id in CARD_INFO_MAP:
            info = CARD_INFO_MAP[c_id]
            return info["name"], info["type"], info["hp"]
    except Exception:
        pass
    return f"Carta #{card_id}", "Desconhecido", None

def get_pokemon_at_position(obs_dict, player_idx, area_type, area_idx):
    """Busca dinamicamente o Pokémon que já está em jogo no Ativo ou no Banco."""
    try:
        curr = obs_dict.get("current", {}) if isinstance(obs_dict, dict) else getattr(obs_dict, "current", {})
        players = curr.get("players", []) if isinstance(curr, dict) else getattr(curr, "players", [])
        
        p_idx = player_idx if player_idx is not None else (curr.get("yourIndex", 0) if isinstance(curr, dict) else getattr(curr, "yourIndex", 0))
        
        if len(players) > p_idx:
            p = players[p_idx]
            area_val = int(area_type) if area_type is not None else 4
            
            # area 4 = active Spot
            if area_val == 4 or str(area_type).upper() in ("ACTIVE", "AREATYPE.ACTIVE"):
                active_list = p.get("active", []) if isinstance(p, dict) else getattr(p, "active", [])
                if active_list and active_list[0]:
                    pkmn = active_list[0]
                    p_id = pkmn.get("id") if isinstance(pkmn, dict) else getattr(pkmn, "id", None)
                    p_hp = pkmn.get("hp") if isinstance(pkmn, dict) else getattr(pkmn, "hp", 0)
                    p_max = pkmn.get("maxHp") if isinstance(pkmn, dict) else getattr(pkmn, "maxHp", 0)
                    name, _, _ = get_card_info(p_id, fallback="Pokémon Ativo")
                    return name, p_hp, p_max
            
            # area 5 = bench Spot
            elif area_val == 5 or str(area_type).upper() in ("BENCH", "AREATYPE.BENCH"):
                bench_list = p.get("bench", []) if isinstance(p, dict) else getattr(p, "bench", [])
                idx = int(area_idx) if area_idx is not None else 0
                if 0 <= idx < len(bench_list) and bench_list[idx]:
                    pkmn = bench_list[idx]
                    p_id = pkmn.get("id") if isinstance(pkmn, dict) else getattr(pkmn, "id", None)
                    p_hp = pkmn.get("hp") if isinstance(pkmn, dict) else getattr(pkmn, "hp", 0)
                    p_max = pkmn.get("maxHp") if isinstance(pkmn, dict) else getattr(pkmn, "maxHp", 0)
                    name, _, _ = get_card_info(p_id, fallback=f"Pokémon do Banco #{idx+1}")
                    return name, p_hp, p_max
    except Exception:
        pass
    return None, None, None

def resolve_target_pokemon_str(obs_dict, player_idx, area_type, area_idx):
    name, hp, max_hp = get_pokemon_at_position(obs_dict, player_idx, area_type, area_idx)
    if name:
        return f"[{name}]({hp}/{max_hp} HP)"
    
    # fallback contextual se o campo ainda não registrou
    area_val = int(area_type) if area_type is not None else 4
    if area_val == 4:
        return "Pokémon Ativo"
    return f"Pokémon do Banco #{int(area_idx)+1 if area_idx is not None else 1}"

def get_active_info_str(obs_dict, player_idx):
    name, hp, max_hp = get_pokemon_at_position(obs_dict, player_idx, 4, 0)
    if name:
        return f"{name}({hp}/{max_hp} HP)"
    return "Pokémon Ativo(?/? HP)"

def describe_option_rich(opt_data, select_data, obs_dict, player_idx) -> str:
    """Narra os movimentos com tratamento inteligente de cartas ocultas e posições de mesa."""
    if not opt_data:
        return "Passou o turno"

    # propriedades da opção
    if isinstance(opt_data, dict):
        t_val = opt_data.get("type", -1)
        card_id = opt_data.get("cardId")
        attack_id = opt_data.get("attackId")
        area = opt_data.get("area")
        in_play_area = opt_data.get("inPlayArea")
        in_play_index = opt_data.get("inPlayIndex")
        target_player = opt_data.get("playerIndex")
    else:
        t_val = getattr(opt_data, "type", -1)
        if hasattr(t_val, "value"):
            t_val = t_val.value
        card_id = getattr(opt_data, "cardId", None)
        attack_id = getattr(opt_data, "attackId", None)
        area = getattr(opt_data, "area", None)
        in_play_area = getattr(opt_data, "inPlayArea", None)
        in_play_index = getattr(opt_data, "inPlayIndex", None)
        target_player = getattr(opt_data, "playerIndex", None)

    # propriedades do contexto de seleção
    if isinstance(select_data, dict):
        ctx_val = select_data.get("context", 0)
        ctx_card = select_data.get("contextCard")
        eff_card = select_data.get("effect")
    else:
        ctx_val = getattr(select_data, "context", 0)
        if hasattr(ctx_val, "value"):
            ctx_val = ctx_val.value
        ctx_card = getattr(select_data, "contextCard", None)
        eff_card = getattr(select_data, "effect", None)

    # identifica a carta que gerou o efeito
    source_card_id = None
    if ctx_card:
        source_card_id = ctx_card.get("id") if isinstance(ctx_card, dict) else getattr(ctx_card, "id", None)
    elif eff_card:
        source_card_id = eff_card.get("id") if isinstance(eff_card, dict) else getattr(eff_card, "id", None)
    
    source_name, _, _ = get_card_info(source_card_id, fallback="") if source_card_id else ("", "", None)

    # informações da carta da opção
    c_name, c_type, c_hp = get_card_info(card_id, fallback="Carta Oculta")

    if t_val == 13: # ATTACK
        atk_name = ATTACK_NAME_MAP.get(attack_id, f"Ataque #{attack_id}") if attack_id else "Ataque Principal"
        p1_str = get_active_info_str(obs_dict, player_idx)
        p2_str = get_active_info_str(obs_dict, 1 - player_idx)
        return f"⚔️ {p1_str} atacou com [{atk_name}] o {p2_str}"

    elif t_val == 8: # ATTACH
        target_str = resolve_target_pokemon_str(obs_dict, target_player or player_idx, in_play_area, in_play_index)
        energy_name = c_name if c_name != "Carta Oculta" else "Carta de Energia"
        return f"⚡ Ligou a energia [{energy_name}] no {target_str}"

    elif t_val == 7: # PLAY
        if "Pokémon" in c_type:
            hp_str = f"({c_hp} HP)" if c_hp else ""
            return f"🃏 Baixou no Banco o Pokémon [{c_name}]{hp_str}"
        elif c_type in ("Supporter", "Item", "Stadium", "Pokémon Tool"):
            type_pt = {"Supporter": "Apoiador", "Item": "Item", "Stadium": "Estádio", "Pokémon Tool": "Ferramenta Pokémon"}.get(c_type, "Treinador")
            return f"🃏 Jogou a carta de Treinador ({type_pt}) [{c_name}]"
        else:
            name_str = c_name if c_name != "Carta Oculta" else "Carta de Treinador/Pokémon"
            return f"🃏 Jogou da mão a carta [{name_str}]"

    elif t_val == 9: # EVOLVE
        target_str = resolve_target_pokemon_str(obs_dict, target_player or player_idx, in_play_area, in_play_index)
        hp_str = f"({c_hp} HP)" if c_hp else ""
        evo_name = c_name if c_name != "Carta Oculta" else "Pokémon Evoluído"
        return f"🧬 Evoluiu o {target_str} para [{evo_name}]{hp_str}"

    elif t_val == 10: # ABILITY
        return f"✨ Usou a Habilidade da carta [{c_name if c_name != 'Carta Oculta' else 'Pokémon'}]"

    elif t_val == 3: # CARD SELECTION
        area_val = int(area) if area is not None else 0
        
        # AreaType.PRIZE = 6 ou Context.TO_PRIZE = 11
        if area_val == 6 or ctx_val == 11:
            prize_str = c_name if c_name != "Carta Oculta" else "Carta de Prêmio Fechada"
            return f"🏆 PEGOU A CARTA DE PRÊMIO [{prize_str}]"
            
        elif ctx_val == 1: # SETUP_ACTIVE_POKEMON
            pkmn_name = c_name if c_name != "Carta Oculta" else "Pokémon Básico Inicial"
            hp_str = f"({c_hp} HP)" if c_hp else ""
            return f"📌 Colocou [{pkmn_name}]{hp_str} como Pokémon Ativo Inicial"
            
        elif ctx_val == 2: # SETUP_BENCH_POKEMON
            pkmn_name = c_name if c_name != "Carta Oculta" else "Pokémon Básico Inicial"
            hp_str = f"({c_hp} HP)" if c_hp else ""
            return f"📌 Colocou [{pkmn_name}]{hp_str} no Banco no Setup"
            
        elif ctx_val in (3, 4): # SWITCH / TO_ACTIVE
            target_str = resolve_target_pokemon_str(obs_dict, player_idx, area_val, 0)
            return f"🔄 Promoveu {target_str} para o Campo Ativo"
            
        elif ctx_val == 7: # TO_HAND
            source_str = f" por efeito de [{source_name}]" if source_name else ""
            card_str = c_name if c_name != "Carta Oculta" else "Carta do Deck/Descarte"
            return f"📥 Selecionou a carta [{card_str}] para a Mão{source_str}"
            
        elif ctx_val == 8: # DISCARD
            source_str = f" por efeito de [{source_name}]" if source_name else ""
            card_str = c_name if c_name != "Carta Oculta" else "Carta da Mão"
            return f"🗑️ Descartou a carta [{card_str}]{source_str}"
            
        elif ctx_val == 25: # EFFECT_TARGET
            target_str = resolve_target_pokemon_str(obs_dict, player_idx, in_play_area, in_play_index)
            return f"🎯 Selecionou {target_str} como alvo da Habilidade/Efeito de [{source_name}]"
            
        else:
            source_str = f" por efeito de [{source_name}]" if source_name else ""
            card_str = c_name if c_name != "Carta Oculta" else "Carta Selecionada"
            return f"🎯 Selecionou a carta [{card_str}]{source_str}"

    elif t_val == 12: # RETREAT
        return "🔄 Recuou o Pokémon Ativo para o Banco"

    elif t_val == 14: # END
        return "Passou o turno"

    elif t_val in (1, 2):
        return "Confirmou seleção (Sim/Não)"

    else:
        label = OPTION_TYPE_NAMES.get(int(t_val), f"AÇÃO_{t_val}")
        card_str = c_name if c_name != "Carta Oculta" else ""
        return f"{label} [{card_str}]" if card_str else label

def run_rich_narrative_benchmark(num_matches=3):
    print("🚀 Executando Benchmark com Tratamento de Campo e Cartas Ocultas...")
    deck = read_deck_csv(DECK_PATH)
    narrative_logs = []

    for idx in range(num_matches):
        res = battle_start(deck, deck)
        obs_dict = res[0] if isinstance(res, tuple) else res
        turn = 0

        while obs_dict and turn < 150:
            turn += 1
            curr_state = obs_dict.get("current", {}) if isinstance(obs_dict, dict) else {}
            p_idx = curr_state.get("yourIndex", 0) if isinstance(curr_state, dict) else getattr(curr_state, "yourIndex", 0)
            
            agent_func = rule_agent if p_idx == 0 else random_agent
            agent_name = "Rule-Based (P1)" if p_idx == 0 else "Random (P2)"
            
            action_indices = agent_func(obs_dict)
            
            select_data = obs_dict.get("select", {}) if isinstance(obs_dict, dict) else getattr(obs_dict, "select", {})
            options = select_data.get("option", []) if isinstance(select_data, dict) else getattr(select_data, "option", [])
            
            descriptions = []
            for a_idx in action_indices:
                if 0 <= a_idx < len(options):
                    descriptions.append(describe_option_rich(options[a_idx], select_data, obs_dict, p_idx))

            narration = " + ".join(descriptions) if descriptions else "Passou o turno"

            narrative_logs.append({
                "Partida": idx + 1,
                "Turno": turn,
                "Jogador": agent_name,
                "Narração": narration
            })

            try:
                obs_dict = battle_select(action_indices)
                res_val = obs_dict.get("current", {}).get("result", -1) if isinstance(obs_dict, dict) else -1
                if res_val != -1:
                    w_str = "Rule-Based (P1)" if res_val == 0 else "Random (P2)"
                    narrative_logs.append({
                        "Partida": idx + 1,
                        "Turno": turn,
                        "Jogador": "SISTEMA",
                        "Narração": f"🏆 FIM DA PARTIDA! Vencedor: {w_str} (Motivo: Nocaute do Pokémon Ativo sem reserva no banco / 0 cartas de Prêmio restantes)."
                    })
                    break
            except Exception:
                break

        battle_finish()

    df = pd.DataFrame(narrative_logs)
    df.to_csv("narrativa_detalhada_partida.csv", index=False)
    print("✅ Narração exportada com sucesso para 'narrativa_detalhada_partida.csv'!")

if __name__ == "__main__":
    run_rich_narrative_benchmark(3)