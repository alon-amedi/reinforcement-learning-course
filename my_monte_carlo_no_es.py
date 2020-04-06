from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from grid_world_utils import *


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

    # print("playing game")
    state = (2, 0)
    action = get_random_action(policy[state], POSSIBLE_ACTIONS)
    grid.set_state(state)
    reward = 0
    # Note that the triplets are (s(t), a(t), r(t)) -  where r(t) is the result of
    # doing action=a(t-1) when in state=s(t-1)
    # states_actions_rewards = [(state, action, 0)]

    states_actions_rewards = [(state, action, reward)]
    while True:
        reward = grid.move(action)
        state = grid.current_state()

        if grid.game_over():
            states_actions_rewards.append((state, None, reward))
            break
        else:
            action = get_random_action(policy[state], POSSIBLE_ACTIONS)
            states_actions_rewards.append((state, action, reward))
    # Calculate returns
    states_actions_rewards.reverse()
    G = 0
    states_actions_returns = []

    for i, (s, a, r) in enumerate(states_actions_rewards):
        if i > 0:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA*G

    states_actions_returns.reverse()
    return states_actions_returns


if __name__ == "__main__":

    N = 5000
    # grid = standard_grid()
    grid = negative_grid()

    policy = {s: np.random.choice(POSSIBLE_ACTIONS) for s in grid.actions.keys()}

    returns = defaultdict(list)
    Q = defaultdict(lambda: defaultdict(float))

    deltas = []
    for t in range(N):

        if t % 100 == 0:
            print(f"Iteration: {t}")

        states_actions_returns = play_game(grid, policy)
        already_seen = defaultdict(bool)
        biggets_change = float("-inf")

        for s, a, G in states_actions_returns:
            s_a = (s, a)
            if s_a not in already_seen: # according to the first-visit method for MC policy evaluation
                old_q = Q[s][a]
                returns[s_a].append(G)
                Q[s][a] = float(np.mean(returns[s_a]))
                biggets_change = max(biggets_change, np.abs(old_q - Q[s][a]))
                already_seen[s_a] = True
        deltas.append(biggets_change)

        # updating policy

        # for s in policy.keys():
        for s in Q.keys():
            policy[s] = sorted(Q[s].items(), key=lambda x: x[1])[-1][0]

    print("Final policy:")
    print_policy(policy, grid)

    plt.plot(deltas)
    plt.title("Deltas per iteration")
    plt.show()
    # Calculate V(s)
    V = {s: sorted(qs.items(), key=lambda x: x[1])[-1][1] for s, qs in Q.items()}
    print("Final values:")
    print_values(V, grid)



