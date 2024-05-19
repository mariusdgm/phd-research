import numpy as np
from collections import Counter


def get_frequency_scaling(transitions):
    transitions_for_counting = [
        (s, a, ns, r, int(d)) for s, a, ns, r, d, _ in transitions
    ]
    transition_counts = Counter(transitions_for_counting)

    # Calculate expected frequency under uniform distribution
    N_total = len(transitions)
    N_unique = len(set(transitions_for_counting))
    expected_frequency = N_total / N_unique

    # Compute scaling factor relative to uniform distribution
    inverse_frequency_scaling = {
        t: expected_frequency / count for t, count in transition_counts.items()
    }
    return inverse_frequency_scaling


def normalize_frequencies(transitions):
    unique_transitions = list(set(transitions))
    total_size = len(transitions)
    num_unique = len(unique_transitions)

    # Calculate the ideal uniform count for each transition
    ideal_count = total_size // num_unique

    # Fill the dataset up to the ideal count
    normalized_transitions = []
    for transition in unique_transitions:
        normalized_transitions.extend([transition] * ideal_count)

    # Calculate remaining space in the dataset
    remaining_space = total_size - len(normalized_transitions)

    # Fill the remaining space by uniform sampling
    if remaining_space > 0:
        indices = np.random.choice(
            len(unique_transitions), size=remaining_space, replace=False
        )
        additional_transitions = [unique_transitions[i] for i in indices]
        normalized_transitions.extend(additional_transitions)

    # Shuffle the dataset to randomize the order
    np.random.shuffle(normalized_transitions)

    return normalized_transitions