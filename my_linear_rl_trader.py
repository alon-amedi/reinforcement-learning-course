import os
import itertools
import argparse
from datetime import datetime
import pickle as pkl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Let's use AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
def get_data():
    # returns a T x 3 list of stock prices
    # each row is a different stock
    # 0 = AAPL
    # 1 = MSI
    # 2 = SBUX
    df = pd.read_csv('aapl_msi_sbux.csv')
    return df.values


def get_scaler(env):
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here

    states = []
    for i in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


class LinearModel:

    def __init__(self, input_dim, n_actions):
        self.W = np.random.randn(input_dim, n_actions) / np.sqrt(input_dim)
        self.b = np.zeros(n_actions)

        # for the momentum
        self.vW = 0
        self.vb = 0

        self.losses = []

    def predict(self, x):
        """
        apply the linear model
        :param x: a (NxD) matrix
        :return:
        """
        assert len(x.shape) == 2, "Assertion of x being 2 dimentional array failed"
        return x.dot(self.W) + self.b

    def sgd(self, x, y, learning_rate=0.01, momentum=0.9):
        """
        Apply one step of gradient descent
        :param x: feature vector
        :param y: target value
        :param learning_rate:
        :param momentum:
        :return:
        """
        assert len(x.shape) == 2, "Assertion of x being 2 dimentional array failed"

        # we'll use #samples * # outputs to average over the sum of errors, not just #samples
        num_of_values = np.prod(y.shape)

        y_hat = self.predict(x)

        # multiplying by 2 to calculate the exact gradient (2 comes from the power in the squared error
        gW = 2 * x.T.dot(y_hat - y) / num_of_values
        gb = 2 * (y_hat - y).sum() / num_of_values

        # update the update values
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update parameters
        self.W += self.vW
        self.b += self.vb

        # log the current loss to losses
        self.losses.append(np.mean((y_hat - y) ** 2))

    def load_weights(self, filepath):
        """
        load a linear model parameterd
        :param filepath:
        :return:
        """
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)


class MultiStockEnv:
    """
    A stock trading env of #stocks = 3 stocks consisting of:
        State - a vector of size 7 (#stocks * 2 + 1)
            - The first 3 elements are number of stocks owned
            - The second 3 elements are prices of stocks owned
            - Last element is the cash left
        Action - a categorical variable with 27 (#stocks * #actions) possibilities
            - for each stock we can do one of (sell, hold, buy) valued by (0, 1, 2)
    """

    def __init__(self, data, initiial_investment=20000):

        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape

        self.initiial_investment = initiial_investment
        self.current_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash = None

        self.action_space = np.arange(3**self.n_stock)
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=3)))

        self.state_dim = 2 * self.n_stock + 1

        self.reset()

    def reset(self):
        """
        Start from the first day having to stocks but all the invested money as cash
        :return:
        """
        self.current_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.current_step]
        self.cash = self.initiial_investment
        return self._get_obs()

    def step(self, action):
        """
        perform the passed action and return the state, reward, is terminal indicator and some info
        :param action: a valid action
        :return: (state, reward, done, info)
        """

        assert action in self.action_space

        # value before performing the action to asses the reward
        prev_val = self._get_val()

        # moving to the next day
        self.current_step += 1
        self.stock_price = self.stock_price_history[self.current_step]

        # perform trade
        self._trade(action)

        # calc returned values
        current_val = self._get_val()
        state = self._get_obs()
        reward = current_val - prev_val
        done = self.current_step == self.n_step - 1
        info = {'current_value': current_val}

        return state, reward, done, info

    def _get_obs(self):
        """
        get the observation / state
        :return: a vector of #stocks * 2 + 1 length
        """
        return np.concatenate([self.stock_owned, self.stock_price, np.array([self.cash])])

    def _get_val(self):
        """
        :return: the value of our portfolio + cach
        """
        return self.stock_owned.dot(self.stock_price) + self.cash

    def _trade(self, action_idx):
        """
        make the trade according to action_idx from the self.action_list
        this will do all the sell operations before the buy operations
        :param action_idx: a integer that represents the index of the action in self.action_list we should act upon
        :return:
        """

        action_vec = self.action_list[action_idx]

        sell_idx = [i for i, a in enumerate(action_vec) if a == 0]
        buy_idx = [i for i, a in enumerate(action_vec) if a == 2]

        # we are simplifying by selling all the stocks of a stock we want to sell
        for s in sell_idx:
            self.cash += self.stock_price[s] * self.stock_owned[s]
            self.stock_owned[s] = 0

        # buying strategy is to buy one of each stock we wish to buy until we have no sufficient funds
        if buy_idx:
            can_buy = True
            while can_buy:
                for b in buy_idx:
                    if self.cash >= self.stock_price[b]:
                        self.cash -= self.stock_price[b]
                        self.stock_owned[b] += 1 # one at a time
                    else:
                        can_buy = False


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0 # chance for exploring
        self.minimal_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)

    def act(self, state):

        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)[0]
        return np.argmax(act_values)

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * (reward + np.amax(self.model.predict(next_state), axis=1))

        # we need target for all actions but we don't want to apply the gradient descent on actions that aren't taken
        # so we set the target of non taken actions to zero
        target_full = self.model.predict(state)
        target_full[0, action] = target

        # perform one step of learning
        self.model.sgd(state, target_full)

        # decide whether to continue
        if self.epsilon > self.minimal_epsilon:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def play_one_episode(agent, env, is_train):

    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train:
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return info["current_value"]


if __name__ == "__main__":

    print("Starting RL on trading data")
    models_folder = "linear_rl_trader_models"
    rewards_folder = "linear_rl_trader_rewards"
    n_episodes = 2000
    batch_size = 32
    initial_investment = 20000

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help="either 'train' or 'test'")
    args = parser.parse_args()
    print(f"Running in {args.mode} mode")

    print(f"Creating folders (if needed)")
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    print("Loading data..")
    data = get_data()
    print("Complete!")

    n_timesteps, n_stocks = data.shape
    n_train = n_timesteps // 2

    print("Split to train and test sets")
    train_data = data[:n_train]
    test_data = data[n_train:]

    print("Setting env and agent")
    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = Agent(state_size, action_size)

    print("Creating scaler")
    scaler = get_scaler(env)

    # store the final value of the portfolio (end of episode)
    portfolio_value = []

    if args.mode == 'test':
        print("Test mode - loading trained models")
        # then load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pkl.load(f)

        # remake the env with test data
        env = MultiStockEnv(test_data, initial_investment)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon = 0.01

        # load trained weights
        agent.load(f'{models_folder}/linear.npz')

    # play the game n_episodes times
    print("Running episodes")
    for e in range(n_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, args.mode)
        dt = datetime.now() - t0
        print(f"episode: {e + 1}/{n_episodes}, episode end value: {val:.2f}, duration: {dt}")
        portfolio_value.append(val)  # append episode end portfolio value

    # save the weights when we are done
    if args.mode == 'train':
        # save the DQN
        agent.save(f'{models_folder}/linear.npz')

        # save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pkl.dump(scaler, f)

        # plot losses
        plt.plot(agent.model.losses)
        plt.show()

    # save portfolio value for each episode
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
