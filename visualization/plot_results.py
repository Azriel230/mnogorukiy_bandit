import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os

def set_plot_style():
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (18, 5)
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10

def plot_experiment_results(avg_results: Dict[str, Dict], 
                           n_steps: int,
                           save_path: Optional[str] = None,
                           save_individual: bool = True):
    """
    Строит три графика по результатам эксперимента.
    
    Аргументы:
        avg_results: результаты от BanditExperiment.get_average_results()
        n_steps: горизонт (T)
        save_path: если указан, сохраняет общий график в файл
        save_individual: если True, сохраняет каждый график отдельно
    """
    set_plot_style()
    
    # Создаём папку для графиков, если её нет
    if save_individual or save_path:
        os.makedirs('plots', exist_ok=True)
    
    # Яркая цветовая палитра
    colors = [
        '#1f77b4',  # синий
        '#ff7f0e',  # оранжевый
        '#2ca02c',  # зелёный
        '#d62728',  # красный
        '#9467bd',  # фиолетовый
        '#8c564b',  # коричневый
        '#e377c2',  # розовый
        '#17becf'   # голубой
    ]
    
    # Разные стили линий
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    
    steps = np.arange(1, n_steps + 1)
    
    # ============= ГРАФИК 1: Средняя награда =============
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    for idx, (name, results) in enumerate(avg_results.items()):
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        ax1.plot(steps, results['avg_rewards'], 
                label=name, color=color, linestyle=linestyle, alpha=0.9, linewidth=2)
    
    ax1.set_xlabel('Шаг t')
    ax1.set_ylabel('Средняя награда')
    ax1.set_title('Средняя награда по шагам')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, n_steps)
    
    plt.tight_layout()
    
    if save_individual:
        plt.savefig('plots/average_reward.png', dpi=150, bbox_inches='tight')
        print("✓ График средней награды сохранён в plots/average_reward.png")
    
    # ============= ГРАФИК 2: Доля оптимальных действий =============
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for idx, (name, results) in enumerate(avg_results.items()):
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        ax2.plot(steps, results['avg_optimal'], 
                label=name, color=color, linestyle=linestyle, 
                alpha=0.9, linewidth=1.5)
    
    ax2.set_xlabel('Шаг t')
    ax2.set_ylabel('Доля оптимальных действий')
    ax2.set_title('Доля выбора оптимального действия')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, n_steps)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_individual:
        plt.savefig('plots/optimal_action_fraction.png', dpi=150, bbox_inches='tight')
        print("✓ График доли оптимальных действий сохранён в plots/optimal_action_fraction.png")
    
    # ============= ГРАФИК 3: Накопленное сожаление =============
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    for idx, (name, results) in enumerate(avg_results.items()):
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        
        # Считаем накопленное сожаление правильно
        cumulative_regret = np.mean(np.cumsum(results['regret_raw'], axis=1), axis=0)
        
        ax3.plot(steps, cumulative_regret, label=name, color=color, 
                linestyle=linestyle, alpha=0.9, linewidth=2)
    
    ax3.set_xlabel('Шаг t')
    ax3.set_ylabel('Накопленное сожаление')
    ax3.set_title('Общее сожаление (сумма)')
    ax3.legend(loc='upper left', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, n_steps)
    
    plt.tight_layout()
    
    if save_individual:
        plt.savefig('plots/cumulative_regret.png', dpi=150, bbox_inches='tight')
        print("✓ График накопленного сожаления сохранён в plots/cumulative_regret.png")
    
    # ============= ОБЩИЙ ГРАФИК (все три вместе) =============
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Средняя награда
    ax = axes[0]
    for idx, (name, results) in enumerate(avg_results.items()):
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        ax.plot(steps, results['avg_rewards'], 
                label=name, color=color, linestyle=linestyle, alpha=0.9)
    ax.set_xlabel('Шаг t')
    ax.set_ylabel('Средняя награда')
    ax.set_title('Средняя награда по шагам')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_steps)
    
    # 2. Доля оптимальных действий
    ax = axes[1]
    for idx, (name, results) in enumerate(avg_results.items()):
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        ax.plot(steps, results['avg_optimal'], 
                label=name, color=color, linestyle=linestyle, alpha=0.9, linewidth=1.5)
    ax.set_xlabel('Шаг t')
    ax.set_ylabel('Доля оптимальных действий')
    ax.set_title('Доля выбора оптимального действия')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_steps)
    ax.set_ylim(0, 1.05)
    
    # 3. Накопленное сожаление
    ax = axes[2]
    for idx, (name, results) in enumerate(avg_results.items()):
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        cumulative_regret = np.mean(np.cumsum(results['regret_raw'], axis=1), axis=0)
        ax.plot(steps, cumulative_regret, label=name, color=color, 
                linestyle=linestyle, alpha=0.9, linewidth=2)
    ax.set_xlabel('Шаг t')
    ax.set_ylabel('Накопленное сожаление')
    ax.set_title('Общее сожаление (сумма)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_steps)
    
    plt.suptitle('Сравнение стратегий в задаче 10-рукого бандита', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Общий график сохранён в {save_path}")
    elif save_individual:
        plt.savefig('plots/combined_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Общий график сохранён в plots/combined_comparison.png")
    
    plt.show()
    
    # Закрываем все фигуры, чтобы освободить память
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig)

def plot_regret_comparison(avg_results: Dict[str, Dict],
                          n_steps: int,
                          save_path: Optional[str] = None):
    """
    Строит отдельный график сожалений для более детального сравнения.
    """
    set_plot_style()
    
    plt.figure(figsize=(14, 7))
    
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#17becf'
    ]
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    steps = np.arange(1, n_steps + 1)
    
    for idx, (name, results) in enumerate(avg_results.items()):
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        
        if 'regret_raw' in results:
            cumulative_regret = np.mean(np.cumsum(results['regret_raw'], axis=1), axis=0)
        else:
            cumulative_regret = results.get('cumulative_regret', 
                                           np.cumsum(results['avg_regret']))
        
        plt.plot(steps, cumulative_regret, label=name, color=color, 
                linestyle=linestyle, linewidth=2.5, alpha=0.9)
    
    plt.xlabel('Шаг t')
    plt.ylabel('Накопленное сожаление')
    plt.title('Сравнение стратегий по накопленному сожалению')
    plt.legend(loc='upper left', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, n_steps)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def plot_learning_curves(avg_results: Dict[str, Dict],
                        n_steps: int,
                        save_path: Optional[str] = None):
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#17becf'
    ]
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    steps = np.arange(1, n_steps + 1)
    
    # Вероятность выбора оптимального действия
    ax = axes[0]
    window = 50
    
    for idx, (name, results) in enumerate(avg_results.items()):
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        optimal_smooth = np.convolve(results['avg_optimal'], 
                                     np.ones(window)/window, mode='same')
        ax.plot(steps, optimal_smooth, label=name, color=color, 
                linestyle=linestyle, linewidth=2)
    
    ax.set_xlabel('Шаг t')
    ax.set_ylabel('P(выбрать оптимум)')
    ax.set_title('Вероятность выбора оптимального действия')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Нормализованное сожаление
    ax = axes[1]
    
    for idx, (name, results) in enumerate(avg_results.items()):
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        
        if 'regret_raw' in results:
            cumulative_regret = np.mean(np.cumsum(results['regret_raw'], axis=1), axis=0)
        else:
            cumulative_regret = results.get('cumulative_regret', 
                                           np.cumsum(results['avg_regret']))
        
        log_steps = np.log(np.maximum(steps, 2))
        normalized_regret = cumulative_regret / log_steps
        ax.plot(steps, normalized_regret, label=name, color=color, 
                linestyle=linestyle, linewidth=2)
    
    ax.set_xlabel('Шаг t')
    ax.set_ylabel('Сожаление / log(t)')
    ax.set_title('Нормализованное сожаление')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()