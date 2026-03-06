import numpy as np
from typing import List

class BernoulliBanditEnv:

    def __init__(self, p_true: List[float]):
        """
        Аргументы:
            p_true: список истинных вероятностей успеха для каждой руки
        """
        self.p_true = np.array(p_true)
        self.k = len(p_true)
        self.optimal_action = np.argmax(self.p_true)
        self.optimal_value = np.max(self.p_true)
    
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)
    
    def step(self, action: int) -> int:
        """
        Возвращает награду 0 или 1.
        """
        reward = np.random.binomial(1, self.p_true[action])
        return reward
    
    @property
    def p_true(self):
        return self._p_true
    
    @p_true.setter
    def p_true(self, value):
        self._p_true = value