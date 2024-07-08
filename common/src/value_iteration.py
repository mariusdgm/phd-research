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
                action_value = 0
                transitions = t_r_dict.get((state, action), [(None, 0, True, 1)])

                # Ensure transitions is always a list
                if not isinstance(transitions, list):
                    transitions = [transitions]

                for transition in transitions:
                    if len(transition) == 4:
                        next_state, reward, done, prob = transition
                    else:
                        next_state, reward, done = transition
                        prob = 1

                    if next_state is None or done:
                        action_value += prob * reward
                    else:
                        action_value += prob * (reward + gamma * V[next_state])

                values.append(action_value)

            V[state] = max(values)
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    policy = {
        state: max(
            actions,
            key=lambda a: sum(
                (
                    prob * (reward + gamma * V.get(next_state, 0))
                    if len(t) == 4
                    else (reward + gamma * V.get(next_state, 0))
                )
                for t in (t_r_dict.get((state, a), [(None, 0, True, 1)]) if isinstance(t_r_dict.get((state, a)), list) else [t_r_dict.get((state, a), (None, 0, True, 1))])
                for next_state, reward, done, prob in ([t] if len(t) == 4 else [t + (1,)])
            ),
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

                action_value = 0
                transitions = t_r_dict.get((state, action), [(None, 0, True)])

                if isinstance(transitions, list) and len(transitions) > 0 and isinstance(transitions[0], tuple):
                    if len(transitions[0]) == 4:
                        for next_state, reward, done, prob in transitions:
                            if next_state is None or done:
                                action_value += prob * reward
                            else:
                                action_value += prob * (reward + gamma * max(Q[next_state].values()))
                    else:
                        for next_state, reward, done in transitions:
                            if next_state is None or done:
                                action_value += reward
                            else:
                                action_value += reward + gamma * max(Q[next_state].values())
                else:
                    next_state, reward, done = transitions
                    if next_state is None or done:
                        action_value += reward
                    else:
                        action_value += reward + gamma * max(Q[next_state].values())

                Q[state][action] = action_value

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
        best_value = float("-inf")
        for action in actions:
            action_value = sum(
                prob * (reward + gamma * V.get(next_state, 0))
                for next_state, reward, _, prob in t_r_dict.get(
                    (state, action), {}
                ).values()
            )
            # Update the best action if a higher value is found, or if the value is tied but the action is smaller
            if action_value > best_value or (
                action_value == best_value
                and (best_action is None or action < best_action)
            ):
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
                    for next_state, reward, done, prob in next_outcomes
                    if not done
                )

                Q[state][action] = q_value
                delta = max(delta, abs(old_q - q_value))

        if delta < theta:
            break

    # Derive policy from Q-values
    policy = {state: max(Q[state], key=Q[state].get) for state in states}

    return policy, Q
