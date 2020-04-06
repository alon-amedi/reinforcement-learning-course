from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from grid_world_utils import *


def get_start_conditions(states, actions):
    start_state_idx = np.random.choice(len(states))
    return states[start_state_idx], np.random.choice(actions)


def play_game_exploring_start(grid, policy, state_action):

    # print("playing game")
    state, action = state_action
    grid.set_state(state)
    reward = 0
    # Note that the triplets are (s(t), a(t), r(t)) -  where r(t) is the result of
    # doing action=a(t-1) when in state=s(t-1)
    # states_actions_rewards = [(state, action, 0)]

    states_actions_rewards = [(state, action, reward)]
    seen_states = set()
    num_steps = 0
    while True:
        num_steps += 1
        old_state = state  # grid.current_state()
        reward = grid.move(action)
        state = grid.current_state()

        # hack to prevent agent from doing an action that keeps it in hte same state (going to the wall)
        # if state == old_state:
        if state in seen_states:
            r = -100 / num_steps
            states_actions_rewards.append((state, None, -100))
            # print("break hack")
            break
        elif grid.game_over():
            states_actions_rewards.append((state, None, reward))
            # print("break game over")
            break
        else:
            action = policy[state]
            # print(f"no break state = {state}, action = {action}")
            states_actions_rewards.append((state, action, reward))
        seen_states.add(state)
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

    N = 2000
    # grid = standard_grid()
    grid = negative_grid()

    policy = fixed_policy_always_win
    policy = {s: np.random.choice(POSSIBLE_ACTIONS) for s in grid.actions.keys()}

    returns = defaultdict(list)
    Q = defaultdict(lambda: defaultdict(float))

    deltas = []
    for t in range(N):

        if t % 100 == 0:
            print(f"Iteration: {t}")
        s_a = get_start_conditions(list(grid.actions.keys()), POSSIBLE_ACTIONS)

        states_actions_returns = play_game_exploring_start(grid, policy, s_a)
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



