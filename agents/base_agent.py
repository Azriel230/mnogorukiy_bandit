import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    
    
    def __init__(self, k: int):
        self.k = k
        self.reset()


    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)

        self.counts = np.zeros(self.k, dtype=int)   # счетчик, сколько раз выбрали каждую руку
        self.values = np.zeros(self.k)              # счетчик, текущие оценки ценности каждой руки

    
    @abstractmethod
    def select_action(self, t: int) -> int:
        pass # абстрактный метод, возвращает индекс выбранного на шаге k действия


    @abstractmethod
    def update(self, action: int, reward: float, t: int) -> None:
        pass #абстрактный метод, обновляет внутреннее состояние агента после получения награды