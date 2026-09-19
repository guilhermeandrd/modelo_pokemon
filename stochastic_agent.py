# stochastic_agent.py
import random
import os
import sys

SAMPLE_DIR = os.environ.get("CABT_SAMPLE_SUBMISSION_DIR", "sample_submission")
if os.path.abspath(SAMPLE_DIR) not in sys.path:
    sys.path.append(os.path.abspath(SAMPLE_DIR))

try:
    from cg.api import to_observation_class
except Exception:
    def to_observation_class(obs):
        return obs

def get_action_priority(opt):
    """Mapeamento estrito e seguro dos tipos de opção."""
    if isinstance(opt, dict):
        opt_type = opt.get("type", -1)
        in_play_area = opt.get("inPlayArea", 4)
    else:
        opt_type = getattr(opt, "type", -1)
        if hasattr(opt_type, "value"):
            opt_type = opt_type.value
        in_play_area = getattr(opt, "inPlayArea", 4)

    try:
        opt_type = int(opt_type)
    except (TypeError, ValueError):
        opt_type = -1

    # prioridades fixas e distantes para evitar ruido destrutivo
    if opt_type == 13:   # ATTACK
        return 10000.0
    elif opt_type == 9:  # EVOLVE
        return 8000.0
    elif opt_type == 8:  # ATTACH
        area_val = int(in_play_area) if in_play_area is not None else 4
        return 6000.0 if area_val == 4 else 4000.0
    elif opt_type == 10: # ABILITY
        return 5000.0
    elif opt_type == 7:  # PLAY
        return 3000.0
    elif opt_type == 3:  # CARD SELECTION / SETUP / PRIZE
        return 2000.0
    elif opt_type in (0, 1): # CONFIRM
        return 1500.0
    elif opt_type == 14: # END TURN
        return 1.0

    return 100.0

def agent(obs_dict, noise_std=2.0):
    """
    Agente Estocástico à prova de falhas com sanitização de tipos para libcg.
    """
    obs = to_observation_class(obs_dict) if isinstance(obs_dict, dict) else obs_dict

    select_data = obs_dict.get("select", {}) if isinstance(obs_dict, dict) else getattr(obs, "select", {})
    options = select_data.get("option", []) if isinstance(select_data, dict) else getattr(select_data, "option", [])

    if not options:
        return []

    min_count = select_data.get("minCount", 1) if isinstance(select_data, dict) else getattr(select_data, "minCount", 1)
    max_count = select_data.get("maxCount", 1) if isinstance(select_data, dict) else getattr(select_data, "maxCount", 1)

    try:
        min_count = int(min_count)
    except Exception:
        min_count = 1

    try:
        max_count = int(max_count)
    except Exception:
        max_count = max(1, min_count)

    scored_options = []
    for idx, opt in enumerate(options):
        base_val = get_action_priority(opt)
        # ruído baixo (noise_std=2.0) apenas para desempate interno
        score = base_val + random.gauss(0, noise_std)
        scored_options.append((score, int(idx)))

    # ordena da maior para a menor pontuação
    scored_options.sort(key=lambda x: x[0], reverse=True)

    # determina quantas ações selecionar garantindo o limite exato do jogo
    num_to_select = max(min_count, min(len(options), max_count))
    selected_indices = [int(idx) for _, idx in scored_options[:num_to_select]]

    # fallback de segurança: se a lista gerada for vazia mas a engine exigia resposta
    if not selected_indices and len(options) >= min_count:
        selected_indices = list(range(min_count))

    return selected_indices