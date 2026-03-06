import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm

from environment.gaussian_bandit_env import GaussianBanditEnv
from agents.greedy_agent import GreedyAgent
from agents.epsilon_greedy_agent import EpsilonGreedyAgent
from agents.ucb_agent import UCBAgent
from agents.gradient_agent import GradientAgent
from agents.thompson_agent import ThompsonSamplingAgent


class BanditExperiment:
    def __init__(self,
                 agents: Dict[str, object],
                 n_runs: int = 200, # количество прогонов (M)
                 n_steps: int = 1000, # горизонт (T)
                 k: int = 10):
        self.agents = agents
        self.n_runs = n_runs
        self.n_steps = n_steps
        self.k = k

        self.results = {}


    def run(self, seed: int = 42) -> Dict[str, Dict]:
        """
        Запускает полный эксперимент.
        """
        print(f"Запуск эксперимента с {self.n_runs} прогонами по {self.n_steps} шагов...")
        print(f"Агенты: {list(self.agents.keys())}\n")
        
        for name in self.agents:
            self.results[name] = {
                'rewards': np.zeros((self.n_runs, self.n_steps)),
                'optimal': np.zeros((self.n_runs, self.n_steps)),
                'regret': np.zeros((self.n_runs, self.n_steps))
            }
        
        for run in tqdm(range(self.n_runs), desc="Прогоны"):
            env = GaussianBanditEnv(k=self.k)
            env.reset(seed=seed + run)
            
            for agent in self.agents.values():
                agent.reset(seed=seed + run)
            
            for t in range(1, self.n_steps + 1):
                for name, agent in self.agents.items():
                    action = agent.select_action(t)
                    
                    reward = env.step(action)
                    
                    agent.update(action, reward, t)
                    
                    self.results[name]['rewards'][run, t-1] = reward
                    self.results[name]['optimal'][run, t-1] = 1 if action == env.a_star else 0
                    self.results[name]['regret'][run, t-1] = env.q_star - env.q_true[action]
        
        print("\nЭксперимент завершён!\n")
        return self.results
    
    def get_average_results(self) -> Dict[str, Dict]:
        avg_results = {}
        
        for name in self.agents:
            avg_results[name] = {
                'avg_rewards': np.mean(self.results[name]['rewards'], axis=0),
                'std_rewards': np.std(self.results[name]['rewards'], axis=0),
                'avg_optimal': np.mean(self.results[name]['optimal'], axis=0),
                'std_optimal': np.std(self.results[name]['optimal'], axis=0),
                'avg_regret': np.mean(self.results[name]['regret'], axis=0),
                'std_regret': np.std(self.results[name]['regret'], axis=0),
                'regret_raw': self.results[name]['regret'],
                'cumulative_regret': np.mean(np.cumsum(self.results[name]['regret'], axis=1), axis=0)
            }
        
        return avg_results
    

    def print_final_statistics(self):
        print("=" * 80)
        print("Результаты (усреднено по {} прогонам)".format(self.n_runs))
        print("=" * 80)
        
        print(f"{'Агент':25} | {'Ср. награда':12} | {'Оптимальных %':12} | {'Сожаление':12}")
        print("-" * 80)
        
        for name in self.agents:
            avg_reward = np.mean(self.results[name]['rewards'])
            optimal_pct = np.mean(self.results[name]['optimal']) * 100
            total_regret = np.sum(np.mean(self.results[name]['regret'], axis=0))
            
            print(f"{name:25} | {avg_reward:12.4f} | {optimal_pct:11.2f}% | {total_regret:12.2f}")
        
        print("=" * 80)