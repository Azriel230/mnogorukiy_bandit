import numpy as np
from agents import Agent

class EpsilonGreedyAgent(Agent):
    
    def __init__(self, k: int, epsilon: float = 0.1):
        self.epsilon = epsilon
        super().__init__(k)

    
    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)


    def select_action(self, t: int) -> int:
        if np.random.random() <= self.epsilon:
            action = np.random.randint(self.k)
            return action
        
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