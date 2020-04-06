import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from grid_world_utils import *
from collections import defaultdict

"""
Note the changes - 
1. 
2. We are no longer decide according to policy convergence but by state-value convergence
3. the convergence flag moved to the iterative_policy_eval finction which now needs to know about the possible actions 
    in order to find max value over actions rather argmax value
"""


def iterative_policy_eval(g, gamma, state_values, policy, th, allowed_actions):
    g_states = g.all_states()
    values = state_values.copy()

    biggest_change = 0
    is_converged = False
    for s in g_states:
        old_v = values[s]
        new_value = float('-inf')
        # V(s) only has value if it's not a terminal state
        if s in policy:
            for a in allowed_actions:
                g.set_state(s)
                r = g.move(a)
                v = r + gamma * values[g.current_state()]
                if v > new_value:
                    new_value = v
            values[s] = new_value
        biggest_change = max(biggest_change, np.abs(old_v - values[s]))

    if biggest_change < th:
        is_converged = True
    print("values for given policy:")
    print_values(values, g)
    print("\n\n")
    return is_converged, values


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
        improved_policy[s] = new_action

    return improved_policy


if __name__ == "__main__":

    # grid = standard_grid()
    grid = negative_grid()
    print_values(grid.rewards, grid)

    # Step 1: initialize policy and state values
    policy = dict(zip(grid.actions.keys(), np.random.choice(POSSIBLE_ACTIONS, len(grid.actions.keys()))))

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
        is_policy_converged, V = iterative_policy_eval(grid, GAMMA, V, policy, TH, POSSIBLE_ACTIONS)
        print("Evaluated V(s):")
        print_values(V, grid)

        if is_policy_converged:
            break

        # Step 3: improve policy - go over each step in the policy
        policy = imporve_policy(grid, policy, V, GAMMA, POSSIBLE_ACTIONS)

    print("Values:")
    print_values(V, grid)
    print("Policy:")
    print_policy(policy, grid)











