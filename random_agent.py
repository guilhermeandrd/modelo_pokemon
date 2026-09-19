# random_agent.py
import random
from cg.api import Observation, to_observation_class

def agent(obs_input) -> list[int]:
    if isinstance(obs_input, dict):
        obs: Observation = to_observation_class(obs_input)
    else:
        obs = obs_input

    if not obs or obs.select is None:
        return [0]

    options = obs.select.option or []
    if not options:
        return [0]

    min_count = obs.select.minCount or 1
    max_count = obs.select.maxCount or 1

    num_to_select = min(max(min_count, 1), max_count, len(options))
    selected = random.sample(range(len(options)), num_to_select)

    return [int(x) for x in selected]