import numpy as np

class BayesBetaAgent:
    """
    Thompson Sampling агент для бернуллиевских бандитов с бета-приором.

    Априор: p_a ~ Beta(alpha0, beta0)
    Апостериор после s успехов и f неудач: Beta(alpha0 + s, beta0 + f)
    """
    
    def __init__(self, k: int, alpha0: float = 1.0, beta0: float = 1.0):
        self.k = k
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.reset()
    
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)

        # Для каждой руки храним количество успехов и неудач
        self.successes = np.zeros(self.k, dtype=int)
        self.failures = np.zeros(self.k, dtype=int)
    
    def select_action(self, t: int) -> int:
        samples = np.zeros(self.k)
        for a in range(self.k):
            # Параметры апостериора для руки a
            alpha = self.alpha0 + self.successes[a]
            beta = self.beta0 + self.failures[a]
            samples[a] = np.random.beta(alpha, beta)
        return np.argmax(samples)
    
    def update(self, action: int, reward: int) -> None:
        if reward == 1:
            self.successes[action] += 1
        else:
            self.failures[action] += 1
    
    @property
    def alpha(self):
        return self.alpha0 + self.successes
    
    @property
    def beta(self):
        return self.beta0 + self.failures