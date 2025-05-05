import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

def objective_function(x):
    return (x - 2)**2 + np.sin(5 * x) * 0.5

def expected_improvement(X, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.min(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

np.random.seed(42)
bounds = np.array([[0.0, 4.0]])

X = np.linspace(bounds[0, 0], bounds[0, 1], 1000).reshape(-1, 1)

X_sample_init = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(5, 1))
Y_sample_init = objective_function(X_sample_init)

X_sample = X_sample_init.copy()
Y_sample = Y_sample_init.copy()

kernel = Matern(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

fig, ax = plt.subplots(figsize=(10, 6))

def update(frame):
    global X_sample, Y_sample

    if frame == 0:
        X_sample = X_sample_init.copy()
        Y_sample = Y_sample_init.copy()

    ax.clear()

    gpr.fit(X_sample, Y_sample)
    mu, std = gpr.predict(X, return_std=True)

    ei = expected_improvement(X, Y_sample, gpr)
    X_next = X[np.argmax(ei)].reshape(1, -1)
    Y_next = objective_function(X_next)

    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))

    ax.plot(X, objective_function(X), 'y--', label='MAE')
    ax.plot(X, mu, 'b-', label='Предсказание алгоритма')
    ax.fill_between(X.ravel(), mu - 1.96 * std, mu + 1.96 * std, alpha=0.3, label='Доверительный интервал')
    ax.scatter(X_sample[:-1], Y_sample[:-1], color='black', label='Прошлые точки', s=50)
    ax.scatter(X_sample[-1], Y_sample[-1], color='red', label='Новая точка', s=100, edgecolor='k')

    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(-1, 6)
    ax.set_xlabel('min_slope')
    ax.set_ylabel('MAE')
    ax.set_title('Байесовская оптимизация')

    ax.legend(title=f'Итерация: {frame + 1}', frameon=False)

ani = FuncAnimation(fig, update, frames=50, interval=300, repeat=True)

plt.tight_layout()
plt.show()
# ani.save('bayes_animation.gif', writer='pillow', fps=5)