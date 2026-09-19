# rule_agent.py
from cg.api import Observation, to_observation_class

def rank_action_text(text: str) -> int:
    t = text.lower()
    if "attack" in t or "atacar" in t:
        return 100
    if "evolve" in t or "evoluir" in t:
        return 80
    if "bench" in t or "play pokemon" in t or "baixar" in t:
        return 70
    if "attach energy" in t or "energia" in t:
        return 60
    if "play" in t or "trainer" in t or "item" in t or "supporter" in t:
        return 40
    if "retreat" in t or "recuar" in t:
        return 20
    if "pass" in t or "end turn" in t or "passar" in t:
        return 1
    return 10

def agent(obs_input) -> list[int]:
    # converte o dicionário retornado pela libcg para a dataclass Observation
    if isinstance(obs_input, dict):
        obs: Observation = to_observation_class(obs_input)
    else:
        obs = obs_input

    # se select for none, o jogo acabou ou nao ha selecao pendente
    if not obs or obs.select is None:
        return [0]

    options = obs.select.option or []
    if not options:
        return [0]

    min_count = obs.select.minCount or 1
    max_count = obs.select.maxCount or 1

    # pontua cada opcao legal do objeto de observacao
    ranked = []
    for idx, opt in enumerate(options):
        text = str(opt.type.name if hasattr(opt.type, 'name') else opt.type)
        score = rank_action_text(text)
        ranked.append((idx, score))

    ranked.sort(key=lambda x: x[1], reverse=True)

    # determina quantas escolhas fazer respeitando minCount e maxCount sem estouro
    num_to_select = min(max(min_count, 1), max_count, len(options))
    selected = [item[0] for item in ranked[:num_to_select]]

    return [int(x) for x in selected]