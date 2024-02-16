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


##############################################################
###### Stochastic case


### Value iteration code for state (V) ###
def policy_evaluation_v_stochastic(states, policy, V, t_r_dict, gamma, theta):
    while True:
        delta = 0
        for state in states:
            v = 0
            action = policy[state]
            transitions = t_r_dict.get((state, action), [])
            for next_state, reward, done, probability in transitions:
                if done:
                    v += probability * reward
                else:
                    v += probability * (reward + gamma * V[next_state])
            delta = max(delta, abs(V[state] - v))
            V[state] = v
        if delta < theta:
            break
    return V


def policy_iteration_v_stochastic(t_r_dict, states, actions, gamma=0.9, theta=1e-6):
    V = {state: 0 for state in states}
    policy = {state: actions[0] for state in states}  # Initial policy

    while True:
        V = policy_evaluation_v_stochastic(states, policy, V, t_r_dict, gamma, theta)
        new_policy = policy_improvement_v_stochastic(
            states, actions, V, t_r_dict, gamma
        )

        if new_policy == policy:
            break
        policy = new_policy

    return policy, V


def policy_improvement_v_stochastic(states, actions, V, t_r_dict, gamma):
    policy = {}
    for state in states:
        action_returns = []
        for action in actions:
            action_return = 0
            transitions = t_r_dict.get((state, action), [])
            for next_state, reward, done, probability in transitions:
                if done:
                    action_return += probability * reward
                else:
                    action_return += probability * (reward + gamma * V[next_state])
            action_returns.append((action, action_return))
        best_action = max(action_returns, key=lambda x: x[1])[0]
        policy[state] = best_action
    return policy


def find_optimal_policy_v_stochastic(t_r_dict, gamma=0.9, theta=1e-6):
    states = list(set(s for s, _ in t_r_dict.keys()))
    actions = list(set(a for _, a in t_r_dict.keys()))

    V = {state: 0 for state in states}
    policy = {state: np.random.choice(actions) for state in states}

    while True:
        V = policy_evaluation_v_stochastic(states, policy, V, t_r_dict, gamma, theta)
        new_policy = policy_improvement_v_stochastic(
            states, actions, V, t_r_dict, gamma
        )

        # Check if the policy has changed
        if new_policy == policy:
            break
        policy = new_policy

    return policy, V


###################################################################
### Policy iteration code for state-action (Q) ###
def policy_improvement_q_stochastic(states, actions, policy, Q):
    new_policy = {}
    policy_stable = True
    for state in states:
        best_action = max(Q[state], key=Q[state].get)
        if policy[state] != best_action:
            policy_stable = False
            new_policy[state] = best_action
        else:
            new_policy[state] = policy[state]

    return new_policy, policy_stable


def policy_iteration_q_stochastic(t_r_dict, states, actions, gamma=0.9, theta=1e-6):
    # Initialize Q-function arbitrarily
    Q = {state: {action: 0 for action in actions} for state in states}
    # Arbitrary initial policy
    policy = {state: np.random.choice(actions) for state in states}

    while True:
        # Evaluate the current policy
        Q = policy_evaluation_q_stochastic(
            states, actions, policy, Q, t_r_dict, gamma, theta
        )
        # Improve the policy based on the evaluated Q-values
        new_policy = policy_improvement_q_stochastic(states, actions, Q)

        # Check if the policy has changed
        if new_policy == policy:
            break  # Policy is stable, exit loop
        policy = new_policy

    return policy, Q


def policy_evaluation_q_stochastic(states, actions, policy, Q, t_r_dict, gamma, theta):
    while True:
        delta = 0
        for state in states:
            for action in actions:
                old_q_value = Q[state][action]
                transitions = t_r_dict.get((state, action), [])
                q_value_sum = 0

                for next_state, reward, done, prob in transitions:
                    if done:
                        q_value_sum += prob * reward  # Directly use the reward if it's a terminal state
                    else:
                        # Use the value of the next state under the current policy, if not terminal
                        q_value_sum += prob * (reward + gamma * Q[next_state][policy[next_state]])
                
                Q[state][action] = q_value_sum
                delta = max(delta, abs(old_q_value - Q[state][action]))

        if delta < theta:
            break
    return Q


def find_optimal_policy_q_stochastic(t_r_dict, gamma=0.9, theta=1e-6):
    states = list(set(s for s, _ in t_r_dict.keys()))
    actions = list(set(a for _, a in t_r_dict.keys()))

    Q = {state: {action: 0 for action in actions} for state in states}
    policy = {state: np.random.choice(actions) for state in states}

    while True:
        Q = policy_evaluation_q_stochastic(
            states, actions, policy, Q, t_r_dict, gamma, theta
        )
        new_policy, policy_stable = policy_improvement_q_stochastic(
            states, actions, policy, Q
        )

        if policy_stable:
            break
        policy = new_policy

    return policy, Q


def random_policy_evaluation_q_stochastic(states, actions, policy, Q, t_r_dict, gamma, theta):
    while True:
        delta = 0
        for state in states:
            for action in actions:
                old_q_value = Q[state][action]
                q_value_sum = 0
                transitions = t_r_dict.get((state, action), [])
                for next_state, reward, done, prob in transitions:
                    if done:
                        q_value_sum += prob * reward
                    else:
                        # For stochastic policy, sum over all actions weighted by their probabilities
                        weighted_sum = sum(
                            policy[state].get(a, 0) * Q[next_state].get(a, 0) for a in actions
                        )
                        q_value_sum += prob * (reward + gamma * weighted_sum)
                Q[state][action] = q_value_sum
                delta = max(delta, abs(old_q_value - Q[state][action]))
        if delta < theta:
            break
    return Q
