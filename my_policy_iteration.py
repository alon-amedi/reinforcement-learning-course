from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from grid_world_utils import *


def iterative_policy_eval(g, gamma, state_values, p, th):
    g_states = g.all_states()
    values = state_values.copy()
    while True:
        biggest_change = 0
        for s in g_states:
            old_v = values[s]
            # V(s) only has value if it's not a terminal state
            if s in p:
                a = p[s]
                g.set_state(s)
                r = g.move(a)
                values[s] = r + gamma * values[g.current_state()]
                current_change = np.abs(old_v - values[s])
                biggest_change = max(biggest_change, current_change)

        if biggest_change < th:
            break
    print("values for given policy:")
    print_values(values, g)
    print("\n\n")
    return values


def imporve_policy(grid, policy, steps_values, gamma, allowed_actions):

    improved_policy = policy.copy()
    is_converged = True
    for s, current_action in improved_policy.items():
        best_val = float('-inf')
        new_action = current_action  # just to have a place holder for best action found

        # iterate over possible actions to find the action that provide the best state value
        for candidate_action in allowed_actions:
            grid.set_state(s)
            r = grid.move(candidate_action)
            v = r + gamma * steps_values[grid.current_state()]
            if v > best_val:
                best_val = v
                new_action = candidate_action
        if new_action != current_action:
            improved_policy[s] = new_action
            is_converged = False

    return is_converged, improved_policy


if __name__ == "__main__":

    # grid = standard_grid()
    grid = negative_grid()
    print_values(grid.rewards, grid)

    # Step 1: initialize policy and state values
    policy = dict(zip(grid.actions.keys(), np.random.choice(ALL_POSSIBLE_ACTIONS, len(grid.actions.keys()))))

    V = defaultdict(int)
    for s in grid.actions:
        V[s] = np.random.random()

    print("Initialized values:")
    print_values(V, grid)
    print("Initialized pplicy:")
    print_policy(policy, grid)

    while True:
        # Step 2: evaluate V(s) for current policy
        print(f"Evaluating V(s) for current policy:")
        print_policy(policy, grid)
        V = iterative_policy_eval(grid, GAMMA, V, policy, TH)
        print("Evaluated V(s):")
        print_values(V, grid)

        # Step 3: improve policy - go over each step in the policy
        is_policy_converged, policy = imporve_policy(grid, policy, V, GAMMA, ALL_POSSIBLE_ACTIONS)
        if is_policy_converged:
            break

    print("Values:")
    print_values(V, grid)
    print("Policy:")
    print_policy(policy, grid)











