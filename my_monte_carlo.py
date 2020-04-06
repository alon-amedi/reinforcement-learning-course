from collections import defaultdict
import numpy as np
from grid_world import standard_grid, negative_grid
from grid_world_utils import *


def play_game(grid, policy, gamma):
    """
    Simulates a game run to evaluate a given policy on a grid.
    we are using the "exploring starts method" to randomly start the game from different states.
    This is required to be able to visit states we are not suppose to visit when using deterministic policy as we do
    which is - go through the upper left path or towards the losing state)

    :param grid: a Grid instance
    :param policy: a dict where keys are states and values are actions
    :param gamma: the discount factor to calculate the returns
    :return: a list of (state, return) pairs
    """

    # exploring start
    actionable_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(actionable_states))
    start_state = actionable_states[start_idx]
    grid.set_state(start_state)

    states_and_rewards = [(grid.current_state(), 0)]  # list of tuples of (state, reward)
    while not grid.game_over():
        a = policy[grid.current_state()]
        r = grid.move(a)
        states_and_rewards.append((grid.current_state(), r))

    g = 0
    states_and_returns = []
    for i, (s, r) in enumerate(reversed(states_and_rewards)):
        if i > 0:
            states_and_returns.append((s, g))
        g = r + (gamma * g)

    states_and_returns.reverse()
    return states_and_returns


if __name__ == "__main__":
    grid = standard_grid()

    ### fixed policy ###
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }

    states = grid.all_states()
    V = defaultdict(float)
    all_returns = defaultdict(list)
    N = 100
    for _ in range(N):
        already_seen = defaultdict(bool)
        states_and_returns = play_game(grid, policy, GAMMA)
        for s, g in states_and_returns:
            # calculate the reward from states_and_rewards by 1st visit method
            if s not in already_seen:
                all_returns[s].append(g)
                V[s] = float(np.mean(all_returns[s]))
                print(V[s])
                already_seen[s] = True

    print("Rewards:")
    print_values(grid.rewards, grid)
    print("Values:")
    print_values(V, grid)
    print("Policy:")
    print_policy(policy, grid)



