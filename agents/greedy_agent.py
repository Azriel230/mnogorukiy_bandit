import numpy as np
from agents import Agent


class GreedyAgent(Agent):

    def __init__(self, k: int):
        super().__init__(k)


    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)

    
    def select_action(self, t: int) -> int:
        max_value = np.max(self.values)
        best_actions = np.where(self.values == max_value)[0]
        action = np.random.choice(best_actions)
        return action
    

    def update(self, action: int, reward: float, t: int) -> None:
        self.counts[action] += 1

        old_value = self.values[action]
        alpha = 1.0 / self.counts[action]
        new_value = old_value + alpha * (reward - old_value)
        
        self.values[action] = new_value