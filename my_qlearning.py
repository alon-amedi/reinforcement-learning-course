from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from grid_world_utils import *
from my_sarsa import get_random_action, argmax_and_max_on_dict


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
    update_counts_sa = {s: {a: 1 for a in ALL_POSSIBLE_ACTIONS} for s in states}

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
        effective_action = argmax_and_max_on_dict(Q[s])[0]
        grid.set_state(s)
        biggest_change = 0
        while not grid.game_over():
            # in Q-learning we have an effective action that actually take place.
            # We select in randomly as Q-learning is an off policy method.

            # Note that we don't use uniform distribution to sample the effective actions
            effective_action = get_random_action(effective_action, ALL_POSSIBLE_ACTIONS, 0.5/t)
            r = grid.move(effective_action)
            s_next = grid.current_state()

            # updating according to action taken
            alpha = ALPHA / update_counts_sa[s][effective_action]
            update_counts_sa[s][effective_action] += 0.005

            max_a, max_Q = argmax_and_max_on_dict(Q[s_next])

            old_qsa = Q[s][effective_action]
            Q[s][effective_action] = old_qsa + alpha * (r + GAMMA * max_Q - old_qsa)
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][effective_action]))

            update_counts[s] += 1
            # Note the update for the action -
            # this is meaningless if we use uniform distribution for sampling in the next iteration
            # but here we use a function that gives an edge to the policy
            s, effective_action = s_next, max_a

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
