import os
from collections import deque
import numpy as np
import random
import pickle
from scipy.stats import entropy
from .frequency_normalization import normalize_frequencies, transform_to_hashable_type


class ReplayBuffer:
    def __init__(self, max_size, state_dim, n_step):
        self.max_size = max_size
        self.state_dim = state_dim
        self.n_step = n_step
        self.buffer = deque(maxlen=self.max_size)
        self.total_appends = 0  # Total number of elements added

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.total_appends += 1

    def sample(self, batch_size):
        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample")

        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        return states, actions, rewards, next_states, dones

    def sample_n_step(self, batch_size, stride=1):
        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample")

        transitions = []
        for _ in range(batch_size):
            start_idx = random.randint(0, len(self) - self.n_step * stride - 1)
            end_idx = start_idx + self.n_step * stride
            samples = self.buffer[start_idx:end_idx:stride]

            state, _, _, _, _ = samples[0]
            _, _, _, next_state, done = samples[-1]

            reward = 0
            for i in range(self.n_step):
                _, action, r, _, _ = samples[i * stride]
                reward += r * pow(0.99, i)

            transitions.append((state, action, reward, next_state, done))

        states, actions, rewards, next_states, dones = zip(*transitions)

        return states, actions, rewards, next_states, dones

    def save(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump((self.buffer, self.total_appends), f)

    def load(self, file_name):
        with open(file_name, "rb") as f:
            self.buffer, self.total_appends = pickle.load(f)

    def calculate_buffer_entropy(self):
        if len(self.buffer) == 0:
            return 0

        examples = [(transition[0], transition[1]) for transition in self.buffer]
        example_strings = [f"{state}_{action}" for state, action in examples]
        unique_examples, counts = np.unique(example_strings, return_counts=True)
        example_entropy = entropy(counts, base=2)

        return example_entropy

    def normalize_replay_buffer(self):
        transitions = list(self.buffer)
        normalized_transitions = normalize_frequencies(transitions)

        normed_buffer = ReplayBuffer(
            self.max_size, self.state_dim, self.n_step
        )
        for transition in normalized_transitions:
            normed_buffer.append(*transition)

        return normed_buffer

    def get_cycle_count(self):
        if self.total_appends < self.max_size:
            return 0
        return (self.total_appends - self.max_size) // self.max_size + 1

    def did_cycle_occur(self):
        return self.total_appends > 0 and self.total_appends % self.max_size == 0

class UniqueReplayBuffer(ReplayBuffer):
    def __init__(self, max_size, state_dim, n_step):
        super().__init__(max_size, state_dim, n_step)
        self.unique_transitions = set()

    def append(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        hashable_transition = tuple(transform_to_hashable_type(elem) for elem in transition)

        if hashable_transition not in self.unique_transitions:
            if len(self.buffer) >= self.max_size:
                oldest_transition = self.buffer.popleft()
                oldest_hashable = tuple(transform_to_hashable_type(elem) for elem in oldest_transition)
                self.unique_transitions.remove(oldest_hashable)

            self.buffer.append(transition)
            self.unique_transitions.add(hashable_transition)
            self.total_appends += 1

    def load(self, file_name):
        with open(file_name, "rb") as f:
            self.buffer, self.total_appends = pickle.load(f)
        self.unique_transitions = set(
            tuple(transform_to_hashable_type(elem) for elem in transition) for transition in self.buffer
        )

    def save(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump((self.buffer, self.total_appends), f)

    def normalize_replay_buffer(self):
        transitions = list(self.buffer)
        normalized_transitions = normalize_frequencies(transitions)

        normed_buffer = UniqueReplayBuffer(
            self.max_size, self.state_dim, self.n_step
        )
        for transition in normalized_transitions:
            normed_buffer.append(*transition)

        return normed_buffer
    
class SparseReplayBuffer(ReplayBuffer):
    def __init__(self, max_size, state_dim, n_step, threshold, normalization_ranges):
        super().__init__(max_size, state_dim, n_step)
        self.threshold = threshold  # Minimum combined normalized distance to consider a transition "different enough"
        self.normalization_ranges = normalization_ranges  # Dictionary of normalization ranges for each transition component

    def append(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        if self.is_different_enough(transition):
            if len(self.buffer) >= self.max_size:
                self.buffer.popleft()  # Remove the oldest transition to make space
            self.buffer.append(transition)
            self.total_appends += 1

    def is_different_enough(self, new_transition):
        for existing_transition in self.buffer:
            distance = self.compute_normalized_distance(existing_transition, new_transition)
            if distance < self.threshold:
                return False
        return True

    def compute_normalized_distance(self, transition1, transition2):
        # Normalize and compute the distance for each component
        state_distance = np.linalg.norm(np.array(transition1[0]) - np.array(transition2[0])) / self.normalization_ranges["state"]
        next_state_distance = np.linalg.norm(np.array(transition1[3]) - np.array(transition2[3])) / self.normalization_ranges["state"]
        action_distance = np.linalg.norm(np.array(transition1[1]) - np.array(transition2[1])) / self.normalization_ranges["action"]
        reward_distance = abs(transition1[2] - transition2[2]) / self.normalization_ranges["reward"]
        done_distance = abs(transition1[4] - transition2[4]) / self.normalization_ranges["done"]

        # Sum the normalized distances to get a total "difference" score
        total_distance = state_distance + next_state_distance + action_distance + reward_distance + done_distance
        return total_distance

    def load(self, file_name):
        with open(file_name, "rb") as f:
            self.buffer, self.total_appends = pickle.load(f)

    def save(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump((self.buffer, self.total_appends), f)

    def normalize_replay_buffer(self):
        transitions = list(self.buffer)
        normalized_transitions = normalize_frequencies(transitions)

        normed_buffer = SparseReplayBuffer(
            self.max_size, self.state_dim, self.n_step, self.threshold, self.normalization_ranges
        )
        for transition in normalized_transitions:
            normed_buffer.append(*transition)

        return normed_buffer