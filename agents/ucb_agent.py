import numpy as np
from agents import Agent

class UCBAgent(Agent):

    def __init__(self, k: int, c: float = 1.4142): # c ~ sqrt(2)
        self.c = c
        super().__init__(k)


    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.t = 0


    def select_action(self, t: int) -> int:
        for action in range(self.k):
            if self.counts[action] == 0:
                return action
            
        ucb_values = np.zeros(self.k)
        for action in range(self.k):
            ucb_values[action] = self.values[action] + self.c * np.sqrt(np.log(t) / self.counts[action])
        
        action = np.argmax(ucb_values)

        return action
    

    def update(self, action: int, reward: float, t: int) -> None:
        self.counts[action] += 1

        old_value = self.values[action]
        alpha = 1.0 / self.counts[action]
        new_value = old_value + alpha * (reward - old_value)
        
        self.values[action] = new_value


    def __str__(self) -> str:
        return f"UCB (c={self.c})"