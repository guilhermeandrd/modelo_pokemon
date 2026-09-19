# q_learning.py
import json
import os
import random
from cg.api import Observation, to_observation_class

Q_TABLE_FILE = "q_table.json"

class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.30):
        self.alpha = alpha      # Taxa de aprendizado
        self.gamma = gamma      # Fator de desconto
        self.epsilon = epsilon  # Taxa de exploração
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.load_q_table()

    def load_q_table(self):
        if os.path.exists(Q_TABLE_FILE):
            try:
                with open(Q_TABLE_FILE, "r") as f:
                    self.q_table = json.load(f)
            except Exception:
                self.q_table = {}

    def save_q_table(self):
        try:
            with open(Q_TABLE_FILE, "w") as f:
                json.dump(self.q_table, f, indent=2)
        except Exception:
            pass

    def extract_obs_data(self, obs: Observation) -> dict:
        """
        Extrai de forma robusta as informações da mesa lendo tanto atributos
        diretos quanto dicionários internos da libcg.
        """
        data = {
            "my_hp": 100,
            "opp_hp": 100,
            "my_prizes": 6,
            "opp_prizes": 6,
            "my_energy": 0
        }
        
        if not obs:
            return data

        # Tenta ler do dicionário bruto se a dataclass encapsular dict
        raw = getattr(obs, "__dict__", {})
        
        # Leitura flexível do estado dos jogadores
        for key in ["p1", "p2", "me", "opponent", "player", "opp"]:
            p_data = getattr(obs, key, raw.get(key, {}))
            if isinstance(p_data, dict):
                if key in ["p1", "me", "player"]:
                    data["my_hp"] = p_data.get("active", {}).get("hp", 100) if isinstance(p_data.get("active"), dict) else 100
                    data["my_prizes"] = p_data.get("prizes_count", p_data.get("prizes", 6))
                else:
                    data["opp_hp"] = p_data.get("active", {}).get("hp", 100) if isinstance(p_data.get("active"), dict) else 100
                    data["opp_prizes"] = p_data.get("prizes_count", p_data.get("prizes", 6))

        # Leitura direta de atributos do topo se existirem
        data["my_hp"] = getattr(obs, "my_active_hp", getattr(obs, "my_hp", data["my_hp"]))
        data["opp_hp"] = getattr(obs, "opp_active_hp", getattr(obs, "opp_hp", data["opp_hp"]))
        data["my_prizes"] = getattr(obs, "my_prizes_remaining", getattr(obs, "my_prizes", data["my_prizes"]))
        data["opp_prizes"] = getattr(obs, "opp_prizes_remaining", getattr(obs, "opp_prizes", data["opp_prizes"]))
        data["my_energy"] = getattr(obs, "my_active_energy", getattr(obs, "my_energy", data["my_energy"]))

        return data

    def get_state_key(self, obs: Observation) -> str:
        if not obs or not hasattr(obs, 'select') or obs.select is None:
            return "terminal"
        
        info = self.extract_obs_data(obs)
        my_hp_bin = max(0, int(info["my_hp"])) // 20
        opp_hp_bin = max(0, int(info["opp_hp"])) // 20
        
        # Quantidade de ações legais disponíveis no turno atual (muda a cada jogada)
        num_options = len(obs.select.option or [])
        
        return f"hp:{my_hp_bin}|opp_hp:{opp_hp_bin}|prizes:{info['my_prizes']}-{info['opp_prizes']}|opts:{num_options}"

    def get_action_category(self, opt) -> str:
        """
        Usa a mesma lógica de inspeção de texto que funciona no seu Rule Agent.
        """
        opt_str = str(opt).lower()
        if hasattr(opt, 'type'):
            opt_str += " " + str(getattr(opt.type, 'name', opt.type)).lower()
            
        if "attack" in opt_str or "atacar" in opt_str:
            return "ATTACK"
        if "evolve" in opt_str or "evoluir" in opt_str:
            return "EVOLVE"
        if "attach" in opt_str or "energia" in opt_str:
            return "ENERGY"
        if "bench" in opt_str or "play" in opt_str or "baixar" in opt_str:
            return "BENCH_TRAINER"
        if "pass" in opt_str or "end" in opt_str or "passar" in opt_str:
            return "PASS"
        return "OTHER"

    def get_q_value(self, state_key: str, action_cat: str) -> float:
        return self.q_table.get(state_key, {}).get(action_cat, 0.0)

    def update_q_value(self, state: str, action_cat: str, reward: float, next_state: str, next_options: list):
        if not state or not action_cat:
            return
            
        old_q = self.get_q_value(state, action_cat)
        
        next_max_q = 0.0
        if next_options and next_state != "terminal":
            next_q_values = [
                self.get_q_value(next_state, self.get_action_category(opt))
                for opt in next_options
            ]
            next_max_q = max(next_q_values) if next_q_values else 0.0

        new_q = old_q + self.alpha * (reward + (self.gamma * next_max_q) - old_q)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        
        self.q_table[state][action_cat] = round(new_q, 4)

    def compute_reward(self, obs: Observation, chosen_cat: str) -> float:
        if not obs:
            return 0.0
        
        info = self.extract_obs_data(obs)
        reward = 0.0
        
        # Diferencia o valor de cada tipo de movimento
        if chosen_cat == "ATTACK":
            reward += 10.0
        elif chosen_cat == "ENERGY":
            reward += 5.0
        elif chosen_cat == "EVOLVE":
            reward += 7.0
        elif chosen_cat == "BENCH_TRAINER":
            reward += 3.0
        elif chosen_cat == "PASS":
            reward -= 2.0  # Penalidade para evitar passar a vez à toa
            
        return reward

    def act(self, obs_input) -> list[int]:
        if isinstance(obs_input, dict):
            obs: Observation = to_observation_class(obs_input)
        else:
            obs = obs_input

        if not obs or obs.select is None:
            if self.last_state and self.last_action:
                self.update_q_value(self.last_state, self.last_action, -10.0, "terminal", [])
                self.save_q_table()
            self.last_state = None
            self.last_action = None
            return [0]

        options = obs.select.option or []
        if not options:
            return [0]

        current_state = self.get_state_key(obs)

        # Associa cada opção à sua categoria
        option_categories = []
        for idx, opt in enumerate(options):
            cat = self.get_action_category(opt)
            option_categories.append((idx, cat))

        # Epsilon-Greedy
        min_count = obs.select.minCount or 1
        max_count = obs.select.maxCount or 1
        num_to_select = min(max(min_count, 1), max_count, len(options))

        if random.random() < self.epsilon:
            selected_indices = random.sample(range(len(options)), num_to_select)
            chosen_cat = option_categories[selected_indices[0]][1]
        else:
            option_scores = []
            for idx, cat in option_categories:
                q_val = self.get_q_value(current_state, cat)
                option_scores.append((idx, q_val, cat))

            option_scores.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [x[0] for x in option_scores[:num_to_select]]
            chosen_cat = option_scores[0][2] if option_scores else "OTHER"

        reward = self.compute_reward(obs, chosen_cat)

        # Atualiza a Q-table com o movimento do passo anterior
        if self.last_state and self.last_action:
            self.update_q_value(self.last_state, self.last_action, reward, current_state, options)

        self.last_state = current_state
        self.last_action = chosen_cat
        self.save_q_table()

        return [int(x) for x in selected_indices]

# Instância global
_q_agent = QLearningAgent()

def agent(obs_input) -> list[int]:
    return _q_agent.act(obs_input)