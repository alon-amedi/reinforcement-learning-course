import numpy as np
from grid_world import standard_grid
from grid_world_utils import *

SMALL_ENOUGH = 1e-3  # threshold for convergence


def init_Vs(states_space):
    return {s: 0 for s in states_space}


def get_policy_uniform_probs(policy):

    probs = {}
    for s, a in policy.items():
        if isinstance(a, str):
            probs[s] = [(a, 1.0)]
        elif isinstance(a, tuple):
            probs[s] = list(zip(a, [1.0 / len(a)] * len(a)))
        else:
            raise ValueError
    return probs


def iterative_policy_evaluation(states, policy, gamma, th):
    V = init_Vs(states)

    # repeat until convergence
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]

            # V(s) only has value if it's not a terminal state
            if s in policy:
                new_v = 0  # we will accumulate the answer
                for a, p_a in policy[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < th:
            break
    return V


if __name__ == '__main__':
    # iterative policy evaluation
    # given a policy, let's find it's value function V(s)
    # we will do this for both a uniform random policy and fixed policy
    # NOTE:
    # there are 2 sources of randomness
    # p(a|s) - deciding what action to take given the state
    # p(s',r|s,a) - the next state and reward given your action-state pair
    # we are only modeling p(a|s) = uniform
    # how would the code change if p(s',r|s,a) is not deterministic?
    grid = standard_grid()

    # states will be positions (i,j)
    # simpler than tic-tac-toe because we only have one "game piece"
    # that can only be at one position at a time
    states = grid.all_states()

    ### uniformly random actions ###
    random_policy_probs = get_policy_uniform_probs(grid.actions)
    V_random_policy = iterative_policy_evaluation(states, random_policy_probs, 1.0, SMALL_ENOUGH)

    print("values for uniformly random actions:")
    print_values(V_random_policy, grid)
    print("\n\n")

    ### fixed policy ###
    policy = fixed_policy_win_or_lose
    print_policy(policy, grid)

    fixed_policy_probs = get_policy_uniform_probs(policy)
    V_fixed_policy = iterative_policy_evaluation(states, fixed_policy_probs, 0.9, SMALL_ENOUGH)

    print("values for fixed policy:")
    print_values(V_fixed_policy, grid)
