import numpy as np


#### Policy iteration code for state (V) ####
def policy_evaluation_v(states, policy, V, t_r_dict, gamma, theta):
    while True:
        delta = 0
        for state in states:
            state_v = V[state]
            action = policy[state]
            next_state, reward, done = t_r_dict.get((state, action), (None, 0, True))

            if next_state is None or done:  # Handling terminal states
                V[state] = reward
            else:
                V[state] = reward + gamma * V[next_state]

            delta = max(delta, abs(state_v - V[state]))

        if delta < theta:
            break
    return V


def policy_improvement_v(states, actions, policy, V, t_r_dict, gamma):
    policy_stable = True
    for state in states:
        old_action = policy[state]

        q_values = []
        for action in actions:
            next_state, reward, done = t_r_dict.get((state, action), (None, 0, True))

            if next_state is None or done:  # Handling terminal states
                q_value = reward
            else:
                q_value = reward + gamma * V[next_state]
            q_values.append(q_value)

        policy[state] = actions[np.argmax(q_values)]

        if old_action != policy[state]:
            policy_stable = False

    return policy, policy_stable


def policy_iteration_v(t_r_dict, policy, V, states, actions, gamma=0.9, theta=1e-6):
    V = policy_evaluation_v(states, policy, V, t_r_dict, gamma, theta)
    policy, policy_stable = policy_improvement_v(
        states, actions, policy, V, t_r_dict, gamma
    )

    return policy, V, policy_stable


def find_optimal_policy_v(t_r_dict, gamma=0.9, theta=1e-6):
    states = list(set([s for s, _ in t_r_dict.keys()]))
    actions = list(set([a for _, a in t_r_dict.keys()]))

    # Initialize value function and policy arbitrarily
    V = {state: 0 for state in states}
    policy = {state: np.random.choice(actions) for state in states}

    policy_stable = False
    while not policy_stable:
        policy, V, policy_stable = policy_iteration_v(
            t_r_dict, policy, V, states, actions, gamma, theta
        )

    return policy, V


#### Policy iteration code for state-action (Q) ####

import numpy as np


def policy_evaluation_q(states, actions, policy, Q, t_r_dict, gamma, theta):
    while True:
        delta = 0
        for state in states:
            for action in actions:
                q_value = Q[state][action]

                next_state, reward, done = t_r_dict.get(
                    (state, action), (None, 0, True)
                )
                if next_state is None or done:  # Handling terminal states
                    Q[state][action] = reward
                else:
                    # Update Q using the value of the next state following the policy
                    Q[state][action] = (
                        reward + gamma * Q[next_state][policy[next_state]]
                    )

                delta = max(delta, abs(q_value - Q[state][action]))

        if delta < theta:
            break
    return Q


def policy_improvement_q(states, actions, policy, Q):
    policy_stable = True
    for state in states:
        old_action = policy[state]

        # Find the action that maximizes Q for the current state
        best_action = max(actions, key=lambda a: Q[state][a])
        policy[state] = best_action

        if old_action != best_action:
            policy_stable = False

    return policy, policy_stable


def policy_iteration_q(t_r_dict, policy, Q, states, actions, gamma=0.9, theta=1e-6):
    Q = policy_evaluation_q(states, actions, policy, Q, t_r_dict, gamma, theta)
    policy, policy_stable = policy_improvement_q(states, actions, policy, Q)

    return policy, Q, policy_stable


def find_optimal_policy_q(t_r_dict, gamma=0.9, theta=1e-6):
    states = list(set([s for s, _ in t_r_dict.keys()]))
    actions = list(set([a for _, a in t_r_dict.keys()]))

    # Initialize Q-function and policy arbitrarily
    Q = {state: {action: 0 for action in actions} for state in states}
    policy = {state: np.random.choice(actions) for state in states}

    policy_stable = False
    while not policy_stable:
        policy, Q, policy_stable = policy_iteration_q(
            t_r_dict, policy, Q, states, actions, gamma, theta
        )

    return policy, Q


### Stochastic case


def policy_evaluation_q_stochastic(states, actions, policy, Q, t_r_dict, gamma, theta):
    while True:
        delta = 0
        for state in states:
            for action in actions:
                old_q_value = Q[state][action]
                next_state, reward, done = t_r_dict.get(
                    (state, action), (None, 0, True)
                )

                if next_state is None or done:
                    Q[state][action] = reward
                else:
                    # Sum over all possible actions in the next state
                    Q[state][action] = reward + gamma * sum(
                        policy[next_state][a] * Q[next_state][a] for a in actions
                    )

                delta = max(delta, abs(old_q_value - Q[state][action]))

        if delta < theta:
            break

    return Q
