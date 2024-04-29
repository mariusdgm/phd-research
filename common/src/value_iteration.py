import numpy as np


def value_iteration_v(t_r_dict, states, actions, gamma=0.9, theta=1e-6):
    V = {state: 0 for state in states}

    while True:
        delta = 0
        for state in states:
            v = V[state]

            # One step lookahead to find the best action value
            values = []
            for action in actions:
                next_state, reward, done = t_r_dict.get(
                    (state, action), (None, 0, True)
                )
                if next_state is None or done:
                    values.append(reward)
                else:
                    values.append(reward + gamma * V[next_state])

            V[state] = max(values)
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    policy = {
        state: max(
            actions,
            key=lambda a: t_r_dict.get((state, a), (None, 0, True))[1]
            + gamma * V.get(t_r_dict.get((state, a), (None, 0, True))[0], 0),
        )
        for state in states
    }

    return policy, V


def value_iteration_q(t_r_dict, states, actions, gamma=0.9, theta=1e-6):
    Q = {state: {action: 0 for action in actions} for state in states}

    while True:
        delta = 0
        for state in states:
            for action in actions:
                q = Q[state][action]

                # One step lookahead to find the new Q value for the current state-action pair
                next_state, reward, done = t_r_dict.get(
                    (state, action), (None, 0, True)
                )
                if next_state is None or done:
                    Q[state][action] = reward
                else:
                    Q[state][action] = reward + gamma * max(
                        [Q[next_state][a] for a in actions]
                    )

                delta = max(delta, abs(q - Q[state][action]))

        if delta < theta:
            break

    policy = {state: max(actions, key=lambda a: Q[state][a]) for state in states}

    return policy, Q

########################################################
### Stochastic case


def value_iteration_v_stochastic(t_r_dict, states, actions, gamma=0.9, theta=1e-6):
    V = {state: 0 for state in states}

    while True:
        delta = 0
        for state in states:
            v = V[state]

            # One step lookahead to find the best action value
            values = []
            for action in actions:
                outcomes = t_r_dict.get((state, action), {})
                value_action = sum(
                    prob * (reward + gamma * V.get(next_state, 0))
                    for next_state, reward, _, prob in outcomes.values()
                )
                values.append(value_action)

            V[state] = max(values)
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    # Policy derivation from the value function, preferring smaller actions in case of ties
    policy = {}
    for state in states:
        best_action = None
        best_value = float('-inf')
        for action in actions:
            action_value = sum(
                prob * (reward + gamma * V.get(next_state, 0))
                for next_state, reward, _, prob in t_r_dict.get((state, action), {}).values()
            )
            # Update the best action if a higher value is found, or if the value is tied but the action is smaller
            if action_value > best_value or (action_value == best_value and (best_action is None or action < best_action)):
                best_value = action_value
                best_action = action
        policy[state] = best_action

    return policy, V

def value_iteration_q_stochastic(t_r_dict, states, actions, gamma=0.9, theta=1e-6):
    Q = {state: {action: 0 for action in actions} for state in states}

    while True:
        delta = 0
        for state in states:
            for action in actions:
                old_q = Q[state][action]

                # Sum over all possible next states given current state and action
                next_outcomes = t_r_dict.get((state, action), [])
                q_value = sum(
                    prob * (reward + gamma * max(Q[next_state].values()))
                    for next_state, reward, done, prob in next_outcomes if not done
                )

                Q[state][action] = q_value
                delta = max(delta, abs(old_q - q_value))

        if delta < theta:
            break

    # Derive policy from Q-values
    policy = {state: max(Q[state], key=Q[state].get) for state in states}

    return policy, Q