import os
import numpy as np
import random
import pickle

from collections import deque
from scipy.stats import entropy
from sklearn.neighbors import KDTree

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
    def __init__(self, max_size, state_dim, n_step, threshold, normalization_ranges, knn_neighbors=10):
        super().__init__(max_size, state_dim, n_step)
        self.threshold = threshold
        self.normalization_ranges = normalization_ranges
        self.knn_neighbors = knn_neighbors
        self.kd_tree = None  # KDTree will be initialized when we have enough transitions
        self.transition_data = []  # List to store flattened transitions for KDTree

    def _normalize_transition(self, transition):
        state, action, reward, next_state, done = transition

        # Normalize each component according to the provided ranges
        norm_state = np.array(state) / self.normalization_ranges["state"]
        norm_action = np.array([action]) / self.normalization_ranges["action"]
        norm_reward = np.array([reward]) / self.normalization_ranges["reward"]
        norm_next_state = np.array(next_state) / self.normalization_ranges["state"]
        norm_done = np.array([done]) / self.normalization_ranges["done"]

        # Flatten and concatenate
        flat_transition = np.concatenate([norm_state.flatten(), norm_action.flatten(), norm_reward.flatten(), norm_next_state.flatten(), norm_done.flatten()])
        
        return flat_transition

    def _update_kd_tree(self):
        if len(self.transition_data) >= self.knn_neighbors:
            self.kd_tree = KDTree(np.array(self.transition_data))

    def append(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        flat_transition = self._normalize_transition(transition)

        if len(self.transition_data) >= self.knn_neighbors:
            dist, _ = self.kd_tree.query([flat_transition], k=self.knn_neighbors)
            if np.mean(dist) < self.threshold:
                return  # Skip adding as it's too similar

        # Add the new transition
        if len(self.buffer) >= self.max_size:
            self.buffer.popleft()  # Remove the oldest transition to make space
            self.transition_data.pop(0)  # Maintain transition data size

        self.buffer.append(transition)
        self.transition_data.append(flat_transition)
        self.total_appends += 1

        # Update KDTree
        self._update_kd_tree()

    def load(self, file_name):
        with open(file_name, "rb") as f:
            self.buffer, self.total_appends = pickle.load(f)
        self.transition_data = [self._normalize_transition(transition) for transition in self.buffer]
        self._update_kd_tree()

    def save(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump((self.buffer, self.total_appends), f)