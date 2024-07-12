import yaml

def create_random_policy(states, actions):
    num_actions = len(actions)
    action_prob = 1.0 / num_actions  # Uniform probability for each action

    policy = {}
    for state in states:
        policy[state] = {action: action_prob for action in actions}

    return policy


def extract_V_from_Q(Q, states):
    return {state: max(Q[state].values()) for state in states if state in Q}


def extract_V_from_Q_for_stochastic_policy(Q, policy, states):
    V = {}
    for state in states:
        V[state] = sum(policy[state][a] * Q[state][a] for a in policy[state])
    return V

def derive_policy_from_q_table(q_table, states, actions):
    """
    Derive the optimal policy from the Q-table.

    Args:
    - q_table (dict): A dictionary where keys are states and values are dictionaries of action-value pairs.
    - states (list): A list of all possible states.
    - actions (list): A list of all possible actions.

    Returns:
    - policy (dict): A dictionary where keys are states and values are the optimal actions.
    """
    policy = {}
    for state in states:
        if state in q_table:  # Check if state exists in the Q-table
            policy[state] = max(actions, key=lambda a: q_table[state][a])
    return policy

