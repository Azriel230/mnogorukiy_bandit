import numpy as np
import matplotlib.pyplot as plt
import os

from agents.greedy_agent import GreedyAgent
from agents.epsilon_greedy_agent import EpsilonGreedyAgent
from agents.ucb_agent import UCBAgent
from agents.gradient_agent import GradientAgent
from agents.thompson_agent import ThompsonSamplingAgent

from experiments.run_experiment import BanditExperiment
from visualization.plot_results import (
    plot_experiment_results,
    plot_regret_comparison,
    plot_learning_curves
)

def main():    
    k = 10                  # количество рук
    T = 1000                # горизонт
    M = 200                 # количество прогонов
    
    print("=" * 60)
    print("ЗАПУСК ЭКСПЕРИМЕНТА: 10-рукий бандит")
    print("=" * 60)
    print(f"Количество рук: {k}")
    print(f"Горизонт (T): {T}")
    print(f"Количество прогонов (M): {M}")
    print("=" * 60)
    
    agents = {
        'Greedy': GreedyAgent(k),
        'ε-Greedy (ε=0.1)': EpsilonGreedyAgent(k, epsilon=0.1),
        'UCB (c=1.4142)': UCBAgent(k, c=1.4142),
        'Gradient (α=0.1)': GradientAgent(k, alpha=0.1),
        'Thompson (τ=1)': ThompsonSamplingAgent(k, tau=1.0),
    }
    
    print(f"\nАгенты в эксперименте ({len(agents)}):")
    for name in agents:
        print(f"  - {name}")
    
    experiment = BanditExperiment(
        agents=agents,
        n_runs=M,
        n_steps=T,
        k=k
    )
    
    results = experiment.run(seed=42)
    
    experiment.print_final_statistics()
    
    avg_results = experiment.get_average_results()
    
    os.makedirs('plots', exist_ok=True)
    
    print("\nПостроение графиков")
    
    plot_experiment_results(
        avg_results, 
        n_steps=T,
        save_path='plots/bandit_comparison.png'
    )
    
    plot_regret_comparison(
        avg_results,
        n_steps=T,
        save_path='plots/regret_comparison.png'
    )
    
    plot_learning_curves(
        avg_results,
        n_steps=T,
        save_path='plots/learning_curves.png'
    )
    
    print("\nЭксперимент завершён")
    print("Графики сохранены в папке 'plots/'")

if __name__ == "__main__":
    main()