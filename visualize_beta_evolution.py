import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import os

from environment.bernoulli_bandit_env import BernoulliBanditEnv
from agents.bayes_beta_agent import BayesBetaAgent

def plot_beta_posteriors(agent, true_p, t, save_path=None):
    """
    Рисует плотности бета-распределений для всех рук в текущий момент t.
    """
    # Создаём сетку значений p от 0 до 1
    p_grid = np.linspace(0, 1, 200)
    
    plt.figure(figsize=(10, 6))
    
    for a in range(agent.k):
        alpha = agent.alpha[a]
        beta_param = agent.beta[a]
        pdf = beta.pdf(p_grid, alpha, beta_param)
        
        # Подпись с параметрами
        label = f'Рука {a}: α={alpha:.1f}, β={beta_param:.1f}'
        plt.plot(p_grid, pdf, label=label, linewidth=2)
        
        # Отметим истинное значение p_a вертикальной линией
        plt.axvline(x=true_p[a], color='gray', linestyle='--', alpha=0.7)
    
    plt.title(f'Апостериорные распределения в момент t = {t}')
    plt.xlabel('p (вероятность успеха)')
    plt.ylabel('Плотность')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
    plt.show()

def run_bernoulli_experiment():
    """
    Запускает эксперимент с бернуллиевским бандитом и строит снимки апостериоров.
    """
    # Параметры
    K = 3  # количество рук
    true_p = [0.2, 0.5, 0.8]  # истинные вероятности успеха
    # Можно также задать [0.2, 0.5, 0.7] как в задании
    
    # Моменты времени для снимков
    snapshots = [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]
    
    # Создаём среду и агента
    env = BernoulliBanditEnv(true_p)
    agent = BayesBetaAgent(k=K, alpha0=1.0, beta0=1.0)
    
    # Сброс с фиксированным seed для воспроизводимости
    env.reset(seed=42)
    agent.reset(seed=42)
    
    # Создаём папку для сохранения графиков
    os.makedirs('beta_plots', exist_ok=True)
    
    # Снимок в момент t=0 (приор)
    plot_beta_posteriors(agent, true_p, t=0, save_path=f'beta_plots/t=000.png')
    
    # Основной цикл
    t = 0
    step = 0
    while t <= max(snapshots):
        if t in snapshots and t > 0:
            plot_beta_posteriors(agent, true_p, t=t, save_path=f'beta_plots/t={t:03d}.png')
        
        # Если дошли до последнего снимка, можно остановиться, но продолжим до конца
        # для полноты картины
        if t == max(snapshots):
            break
        
        # Один шаг взаимодействия
        action = agent.select_action(t+1)  # t не используется, но передаём для совместимости
        reward = env.step(action)
        agent.update(action, reward)
        
        t += 1
    
    # Дополнительно можно показать финальное состояние после всех шагов (например, после 500)
    # Для интереса продлим до 500 шагов и покажем последний снимок
    while t < 500:
        action = agent.select_action(t+1)
        reward = env.step(action)
        agent.update(action, reward)
        t += 1
    
    plot_beta_posteriors(agent, true_p, t=500, save_path='beta_plots/t=500.png')

if __name__ == "__main__":
    run_bernoulli_experiment()