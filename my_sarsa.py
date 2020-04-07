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


def argmax_and_max_on_dict(dt):
    return sorted(dt.items(), key=lambda x: x[1])[-1]


# Note there's not play_game function as we are updating along the game, meaning this is fully online
if __name__ == "__main__":

    # grid = standard_grid()
    grid = negative_grid()
    states = grid.all_states()
    print("rewards:")
    print_values(grid.rewards, grid)
    # Note that when using standard grid there's no punishment / cost on steps. particularly when making an action R
    # from cell (1, 0) into the wall. it's ok keep doing that or ending up in one of the neighboring states until
    # getting out of there. instead we can penalize each movement so the agent will find its way faster.

    Q = {s: {a: 0 for a in ALL_POSSIBLE_ACTIONS} for s in states}
    # let's also keep track of how many times Q[s] has been updated
    update_counts = defaultdict(int)
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

    # t is used for epsilon greedy - notice it'll be increased every 100 iterations by small amount
    t = 1.0
    deltas = []
    N = 10000
    for it in range(N):
        if (it+1) % 100 == 0:
            t += 1e-2
        if it % 2000 == 0:
            print(f"Iteration: {it} , t: {t}")

        s = (2, 0)
        grid.set_state(s)
        a = argmax_and_max_on_dict(Q[s])[0]
        a = get_random_action(a, ALL_POSSIBLE_ACTIONS, epsilon=0.5/t)

        biggest_change = 0
        while not grid.game_over():
            r = grid.move(a)
            s_next = grid.current_state()
            a_next = argmax_and_max_on_dict(Q[s_next])[0]
            a_next = get_random_action(a_next, ALL_POSSIBLE_ACTIONS, 0.5 / t)

            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005
            old_qsa = Q[s][a]
            Q[s][a] = Q[s][a] + alpha * (r + GAMMA * Q[s_next][a_next] - Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

            update_counts[s] += 1
            s, a = s_next, a_next

        deltas.append(biggest_change)
    plt.plot(deltas)
    plt.show()

    # determine policy from Q*
    policy = {s: argmax_and_max_on_dict(Q[s])[0] for s in grid.actions.keys()}
    # find V(s) from Q*
    V = {s: argmax_and_max_on_dict(Q[s])[1] for s in grid.actions.keys()}

    # what's the proportin of time we spent in each state?
    print("update counts:")
    total = np.sum(list(update_counts.values()))
    proportions = {k: v/total for k, v in update_counts.items()}
    print_values(proportions, grid)

    print("values:")
    print_values(V, grid)

    print("policy:")
    print_policy(policy, grid)
