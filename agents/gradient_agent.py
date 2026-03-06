import numpy as np
from agents import Agent

class GradientAgent(Agent):
    def __init__(self, k: int, alpha: float = 0.1):
        self.alpha = alpha
        super().__init__(k)


    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.preferences = np.zeros(self.k)

    
    def _softmax_probabilities(self) -> np.ndarray:
        prefs = self.preferences - np.max(self.preferences) # защита от переполнение exp
        exp_prefs = np.exp(prefs)
        probs = exp_prefs / np.sum(exp_prefs)
        return probs
    

    def select_action(self, t: int) -> int:
        probs = self._softmax_probabilities()
        action = np.random.choice(self.k, p=probs)
        return action
    

    def update(self, action: int, reward: float, t: int) -> None:
        probs = self._softmax_probabilities()
        for a in range(self.k):
            if a == action:
                self.preferences[a] += self.alpha * reward * (1 - probs[a])
            else:
                self.preferences[a] -= self.alpha * reward * probs[a]


    def __str__(self) -> str:
        return f"Gradient (α={self.alpha})"