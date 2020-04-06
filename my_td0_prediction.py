from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from grid_world_utils import *
from grid_world import standard_grid, negative_grid

# The epsilon-soft will be used here to ensure all states are visited.
# If we won't do this (epsilon=0) - we won't have enough values or at all for some states and we want to have a
# proper exploration here


def get_random_action(a, possible_actions, epsilon=0.1):
    """
    use soft-greedy to choose an action to perform
    :param a: policy action
    :param possible_actions: all possible actions from
    :param epsilon: exploration probability, defaults to 0.1
    :return: action to perform
    """
    assert 0.0 < epsilon < 1.0, "epsilon must be in [0, 1]"
    r = np.random.random()
    return a if r <= (1-epsilon) else np.random.choice(list(set(possible_actions).difference(a)))


def play_game(grid, policy):
    """
    this time this is much simpler becuase we don't need to calculate the returns but return a list of states and rewards
    :param grid:
    :param policy:
    :return:
    """
    state = (2, 0)
    grid.set_state(state)
    states_and_rewards = [(state, 0)]
    while not grid.game_over():
        action = get_random_action(policy[state], POSSIBLE_ACTIONS)
        reward = grid.move(action)
        state = grid.current_state()
        states_and_rewards.append((state, reward))
    return states_and_rewards


if __name__ == "__main__":
    # We are only doing the 1st step here - Addressing the prediction problem and evaluating V(s)

    grid = standard_grid()
    policy = fixed_policy_win_or_lose
    N = 1000
    V = defaultdict(float)

    for _ in range(N):
        states_rewards = play_game(grid, policy)
        # we won't do the online update as it makes the code harder
        # the first (s, r) is 0 - it's the start of the game
        # the last (s, r) is of the terminal state and therefor the final reward.
        # the value of the terminal state is by definition 0 and we don't want to caluculate it
        for i, s_r in enumerate(states_rewards[:-1]):
            s, _ = s_r
            s_next, r_next = states_rewards[i+1]
            V[s] = V[s] + ALPHA * (r_next + GAMMA * V[s_next] - V[s])

    print("Values:")
    print_values(V, grid)
    print("Policy:")
    print_policy(policy, grid)
