from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from grid_world_utils import *
from my_iterative_policy_evaluation import iterative_policy_evaluation


def get_expected_v(action_probs, grid, gamma, states_values, current_state):
    mean_v = 0
    for effective_action, p in action_probs.items():
        grid.set_state(current_state)
        r = grid.move(effective_action)
        mean_v += p * (r + gamma * states_values[grid.current_state()])
    return mean_v


def imporve_policy(grid, policy, states_values, gamma, allowed_actions, intent_prob):

    improved_policy = policy.copy()
    is_converged = True
    for s, current_action in improved_policy.items():
        best_val = float('-inf')
        new_action = current_action  # just to have a place holder for best action found

        # iterate over possible actions to find the action that provide the best state value
        for candidate_action in allowed_actions:

            probs = {a: (intent_prob if candidate_action == a else (1-intent_prob) / (len(allowed_actions) - 1))
                     for a in allowed_actions}
            mean_v = get_expected_v(probs, grid, gamma, states_values, s)

            if mean_v > best_val:
                best_val = mean_v
                new_action = candidate_action
        if new_action != current_action:
            improved_policy[s] = new_action
            is_converged = False

    return is_converged, improved_policy


if __name__ == "__main__":

    np.random.seed(123)

    is_windy = False #True # False
    intent_action_prob = 0.5 if is_windy else 1.0

    grid = negative_grid(-0.1)  # standard_grid()
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
        policy_probs = {s: [(aa, intent_action_prob if aa == a else (1-intent_action_prob) / 3) for aa in ALL_POSSIBLE_ACTIONS]
                        for s, a in policy.items()}

        V = iterative_policy_evaluation(grid, policy_probs, GAMMA, TH, V)
        # V = iterative_policy_eval(grid, GAMMA, V, policy, TH, ALL_POSSIBLE_ACTIONS)
        print("Evaluated V(s):")
        print_values(V, grid)

        # Step 3: improve policy - go over each step in the policy
        is_policy_converged, policy = imporve_policy(grid, policy, V, GAMMA, ALL_POSSIBLE_ACTIONS, intent_action_prob)
        if is_policy_converged:
            break

    print("Values:")
    print_values(V, grid)
    print("Policy:")
    print_policy(policy, grid)











