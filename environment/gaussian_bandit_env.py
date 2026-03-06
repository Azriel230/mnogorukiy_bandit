import numpy as np


class GaussianBanditEnv:
    def __init__(self,
                 k: int = 10,
                 q_mean: float = 0.0,
                 q_std: float = 1.0,
                 reward_std: float = 1.0):
        
        self.k = k                      # количество рук бандита
        self.q_mean = q_mean            # среднее для q(a)
        self.q_std = q_std              # стандартное отклонение для q(a)
        self.reward_std = reward_std    # стандартное отклонение награды

        self.q_mean_true_values = None  # истинные q(a)
        self.optimal_action = None      # оптимальное действие a*
        self.optimal_value = None       # оптимальное значение q*


    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        
        self.q_mean_true_values = np.random.standard_normal(self.k)
        self.optimal_action = np.argmax(self.q_mean_true_values)
        self.optimal_value = np.max(self.q_mean_true_values)

    
    def step(self, action: int) -> float:
        reward = np.random.normal(
            loc=self.q_mean_true_values[action],
            scale=self.reward_std
        )
        return reward
    

    @property
    def q_star(self) -> float:
        return self.optimal_value
    

    @property
    def a_star(self) -> int:
        return self.optimal_action
    

    @property
    def q_true(self) -> np.ndarray:
        return self.q_mean_true_values