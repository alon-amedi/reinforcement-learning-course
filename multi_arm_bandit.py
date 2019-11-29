import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class AbstractBandit:

    def __init__(self, mean, initial_val=0):
        self.mean_hat = initial_val
        self.n = 0
        self.mean = mean

    def pull(self):
        return np.random.normal(self.mean)

    def update(self, x):
        self.n += 1
        self.mean_hat = (1 - 1.0 / self.n) * self.mean_hat + x / self.n

    def get_estimates_str(self):
        return "mean_hat = {}".format(self.mean_hat)

    @staticmethod
    @abstractmethod
    def choose_bandit(bandits, epsilon, *args, **kwargs):
        pass


class EpsilonGreedyBandit(AbstractBandit):

    def __init__(self, mean):
        super().__init__(mean)

    # def pull(self):
    #     return np.random.normal(self.mean)

    # def update(self, x):
    #     self.n += 1
    #     self.mean_hat = (1 - 1.0/self.n) * self.mean_hat + x/self.n

    @staticmethod
    def choose_bandit(bandits, epsilon, *args, **kwargs):
        r = np.random.random()
        if r < epsilon:
            return bandits[np.random.choice(len(bandits))]
        else:
            return bandits[np.argmax([b.mean_hat for b in bandits])]


class OivBandit(AbstractBandit):

    def __init__(self, mean, initial_val):
        super().__init__(mean, initial_val)

    @staticmethod
    def choose_bandit(bandits, *args, **kwargs):
        return bandits[np.argmax([b.mean_hat for b in bandits])]


class Ucb1Bandit(AbstractBandit):

    def __init__(self, mean):
        super().__init__(mean)

    @staticmethod
    def choose_bandit(bandits, *args, **kwargs):
        e = 10E-6
        n = np.sum([b.n for b in bandits]) + e
        j = np.argmax([b.mean_hat + np.sqrt(2 * np.log(n) / (b.n + e)) for b in bandits])
        return bandits[j]


class BayesianBandit(AbstractBandit):

    def __init__(self, mean):
        super().__init__(mean)
        # self.m0 = 0
        self.lambda0 = 1
        self.sum_x = 0
        self.tau = 1

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda0) + self.mean_hat

    def pull(self):
        return np.random.randn() + self.mean

    def update(self, x):
        self.n += 1
        self.lambda0 += self.tau  # assuming tau=1
        self.sum_x += x
        self.mean_hat = self.tau * self.sum_x / self.lambda0

    def get_estimates_str(self):
        return "m0={}, lambda0={}".format(self.mean_hat, self.lambda0)

    @staticmethod
    def choose_bandit(bandits, *args, **kwargs):
        j = np.argmax([b.sample() for b in bandits])
        return bandits[j]


def over_all_experiment(true_means, n):
    opt_val = 10
    kw = {'epsilon': 0.05}
    classes_dict = {
        'eg': EpsilonGreedyBandit,
        'ov': OivBandit,
        'ucb1': Ucb1Bandit,
        'bayesian': BayesianBandit
    }

    bandits_dict = {
        'eg': [EpsilonGreedyBandit(m) for m in true_means],
        'ov': [OivBandit(m, opt_val) for m in true_means],
        'ucb1': [Ucb1Bandit(m) for m in true_means],
        'bayesian': [BayesianBandit(m) for m in true_means]
    }

    data = {k: np.empty(n) for k in bandits_dict.keys()}

    for i in range(n):
        for k, v in bandits_dict.items():
            bandit_to_pull = classes_dict[k].choose_bandit(v, **kw)
            result = bandit_to_pull.pull()
            bandit_to_pull.update(result)
            data[k][i] = result

    for k, v in data.items():
        cumulative_average = np.cumsum(v) / (np.arange(n)+1)
        plt.plot(cumulative_average, label=k)
    plt.hlines(true_means, xmin=0, xmax=n)
    plt.xscale('log')
    plt.legend(loc="lower right")
    plt.title("cumulative averages")
    plt.show()

    for k, v in bandits_dict.items():
        print(k, [b.get_estimates_str() for b in v])


def generic_exp(true_means, n, values, bandit_class, **kwargs):
    from inspect import signature

    n_params = len(signature(bandit_class.__init__).parameters)

    bandits_dict = {v: [bandit_class(m, v) if n_params == 3 else bandit_class(m) for m in true_means] for v in values}
    data = {k: np.empty(n) for k in bandits_dict.keys()}

    for i in range(n):
        for k, v in bandits_dict.items():
            bandit_to_pull = bandit_class.choose_bandit(v, *[k])
            result = bandit_to_pull.pull()
            bandit_to_pull.update(result)
            data[k][i] = result

    for k, v in data.items():
        cumulative_average = np.cumsum(v) / (np.arange(n) + 1)
        plt.plot(cumulative_average, label=k)
    plt.hlines(true_means, xmin=0, xmax=n)
    plt.xscale('log')
    plt.legend(loc="lower right")
    plt.title(f"cumulative averages {bandit_class.__name__}")
    plt.show()


if __name__ == "__main__":

    n = 100000
    means = [1.0, 2.0, 3.0]

    epsilons = [0.01, 0.05, 0.1]
    generic_exp(means, n, epsilons, EpsilonGreedyBandit)

    opt_vals = [5, 10, 20]
    generic_exp(means, n, opt_vals, OivBandit)

    generic_exp(means, n, ["ucb"], Ucb1Bandit)

    over_all_experiment(means, n)

