import numpy as np
from agents import Agent

class ThompsonSamplingAgent(Agent):
    def __init__(self, k: int, tau: float = 1.0):
        self.tau = tau
        super().__init__(k)


    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.sum_rewards = np.zeros(self.k)


    def _posterior_parameters(self, action: int) -> tuple[float, float]:
        n = self.counts[action]
        sum_r = self.sum_rewards[action]

        prior_precision = 1.0 / (self.tau ** 2)

        if n == 0:
            return 0.0, self.tau  # prior: N(0, tau^2)
        
        posterior_precision = prior_precision + n
        posterior_mean = sum_r / posterior_precision
        posterior_std = np.sqrt(1.0 / posterior_precision)

        return posterior_mean, posterior_std
    

    def select_action(self, t: int) -> int:
        samples = np.zeros(self.k)

        for action in range(self.k):
            mean, std = self._posterior_parameters(action)
            samples[action] = np.random.normal(mean, std)

        action = np.argmax(samples)
        return action
    

    def update(self, action: int, reward: float, t: int) -> None:
        self.counts[action] += 1
        self.sum_rewards[action] += reward

        old_value = self.values[action]
        alpha = 1.0 / self.counts[action]
        new_value = old_value + alpha * (reward - old_value)
        
        self.values[action] = new_value


    def posterior_pdf(self, action: int, mu_grid: np.ndarray) -> np.ndarray:
        # вычисляет плотность апостериора для визуализации
        mean, std = self._posterior_parameters(action)
        
        pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * \
              np.exp(-0.5 * ((mu_grid - mean) / std) ** 2)
        
        return pdf
    
    
    def __str__(self) -> str:
        return f"Thompson (τ={self.tau})"