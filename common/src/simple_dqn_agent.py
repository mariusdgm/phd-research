import time
import datetime
import torch
import random
import numpy as np
import os
import pickle
from pathlib import Path
from typing import List, Dict
import warnings

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


import gym

from .replay_buffer import ReplayBuffer, UniqueReplayBuffer
from common.src.experiment_utils import seed_everything

from common.src.models import QNET

# Always save: return and number of steps per episode, max q per step of all episodes
# Interesting info to save:
# per training episode: sequence of states (frames), actions, Q(s_k, a) for all a, rewards, and perhaps episode length/number of steps to target (dar asta se poate infera)
# per network checkpoint (ideally for each point in the graph/epoch): network weights
# per target network update: target network weights

# And to make it perfectly reproducible: original and renormalized buffers, minibatches or minibatch indices (but they recoverable from the seed and trajectories)

# save stuff, rerun, make sure agent learns
# if yes, then make a skewed 2-room or whatever problem, try in that problem, control amount of skew, see if there is signal in normalized vs unnormalized


def replace_keys(d, original_key, new_key):
    """
        Recursively iterates through a dictionary and its sub-dictionaries,
        replacing a specified key with a new key.
    Starting training epoch at t = 0
    2024-06-14 14:58:27,941 - 2024Jun14-145252_configs_algorithm=default - INFO - Episode 0 terminated at frame 399 with reward 0
    2024-06-14 14:58:27,953 - 2024Jun14-145252_configs_a"""
    new_dict = {}
    for key, value in d.items():
        # Replace the key if it matches the specified original key
        updated_key = new_key if key == original_key else key

        # Recursively apply this function if the value is another dictionary
        if isinstance(value, dict):
            new_dict[updated_key] = replace_keys(value, original_key, new_key)
        else:
            new_dict[updated_key] = value
    return new_dict


# TODO: (NICE TO HAVE) gpu device at: model, wrapper of environment (in my case it would be get_state...),
# maybe: replay buffer (recommendation: keep on cpu, so that the env can run on gpu in parallel for multiple experiments)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class AgentDQN:
    def __init__(
        self,
        train_env,
        validation_env,
        experiment_output_folder=None,
        experiment_name=None,
        resume_training_path=None,
        save_checkpoints=True,
        save_replay_buffer_cycles=True,
        logger=None,
        config={},
    ) -> None:
        """A DQN agent implementation.

        Args:
            train_env (gym.env): An instantiated gym Environment
            validation_env (gym.env): An instantiated gym Environment that was created with the same
                                    parameters as train_env. Used to be able to do validation epochs and
                                    return to the same training point.
            experiment_output_folder (str, optional): Path to the folder where the training outputs will be saved.
                                                         Defaults to None.
            experiment_name (str, optional): A string describing the experiment being run. Defaults to None.
            resume_training_path (str, optional): Path to the folder where the outputs of a previous training
                                                    session can be found. Defaults to None.
            save_checkpoints (bool, optional): Whether to save the outputs of the training. Defaults to True.
            logger (logger, optional): Necessary Logger instance. Defaults to None.
            config (Dict, optional): Settings of the agent relevant to the models and training.
                                    If none is provided in the input, the agent will automatically build the default settings.
                                    Defaults to {}.
        """

        # assign environments
        self.train_env = train_env
        self.validation_env = validation_env

        self.save_checkpoints = save_checkpoints
        self.save_replay_buffer_cycles = save_replay_buffer_cycles
        self.logger = logger
        self.tensor_board_writer = None
        self.start_state_counts = {}

        # set up path names
        self.experiment_output_folder = experiment_output_folder
        self.experiment_name = experiment_name

        self.model_file_folder = (
            "model_checkpoints"  # models will be saved at each epoch
        )
        self.model_checkpoint_file_basename = "mck"

        if self.experiment_output_folder and self.experiment_name:
            self.replay_buffer_file = os.path.join(
                self.experiment_output_folder, f"{self.experiment_name}_replay_buffer"
            )
            self.train_stats_file = os.path.join(
                self.experiment_output_folder, f"{self.experiment_name}_train_stats"
            )

        self.config = config
        if self.config:
            self.config = replace_keys(self.config, "args_", "args")

        self._load_config_settings(self.config)
        self._init_models(self.config)  # init policy, target and optim

        self.normalize_replay_buffer_freq = config.get(
            "normalize_replay_buffer_freq", False
        )

        # Set initial values related to training and monitoring
        self.t = 0  # frame nr
        self.episodes = 0  # episode nr
        self.policy_model_update_counter = 0

        self.reset_training_episode_tracker()

        self.training_stats = []
        self.validation_stats = []

        # check that all paths were provided and that the files can be found
        if resume_training_path:
            self.load_training_state(resume_training_path)

    def _make_model_checkpoint_file_path(self, experiment_output_folder, epoch_cnt=0):
        """Dynamically build the path where to save the model checkpoint."""
        return os.path.join(
            experiment_output_folder,
            self.model_file_folder,
            f"{self.model_checkpoint_file_basename}_{epoch_cnt}",
        )

    def load_training_state(self, resume_training_path: str):
        """In order to resume training the following files are needed:
        - ReplayBuffer file
        - Training stats file
        - Model weights file (found as the last checkpoint in the models subfolder)
        Args:
            resume_training_path (str): path to where the files needed to resume training can be found
        """

        ### build the file paths
        resume_files = {}

        resume_files["replay_buffer_file"] = os.path.join(
            resume_training_path, f"{self.experiment_name}_replay_buffer"
        )
        resume_files["train_stats_file"] = os.path.join(
            resume_training_path, f"{self.experiment_name}_train_stats"
        )

        # check that the file paths exist
        for file in resume_files:
            if not os.path.exists(resume_files[file]):
                warnings.warn(
                    f"Could not find the file {resume_files[file]} for {file} either because a wrong path was given, or because no training was done for this experiment."
                )
                return False

        # read through the stats file to find what was the epoch for the last recorded state
        self.load_training_stats(resume_files["train_stats_file"])
        self.replay_buffer.load(resume_files["replay_buffer_file"])

        epoch_cnt = len(self.training_stats)

        resume_files["checkpoint_model_file"] = self._make_model_checkpoint_file_path(
            resume_training_path, epoch_cnt
        )
        if not os.path.exists(resume_files["checkpoint_model_file"]):
            warnings.warn(
                f"Could not find the file {resume_files['checkpoint_model_file']} for 'checkpoint_model_file'. Skipping model loading."
            )
            return False

        self.load_models(resume_files["checkpoint_model_file"])

        start_state_counts_file = os.path.join(
            resume_training_path, f"{self.experiment_name}_start_state_counts.pkl"
        )
        if os.path.exists(start_state_counts_file):
            with open(start_state_counts_file, "rb") as f:
                self.start_state_counts = pickle.load(f)
        else:
            self.start_state_counts = {}

        self.logger.info(
            f"Loaded previous training status from the following files: {str(resume_files)}"
        )

    def _load_config_settings(self, config={}):
        """
        Load the settings from config.
        If config was not provided, then default values are used.
        """
        agent_params = config.get("agent_params", {}).get("args", {})

        # setup training configuration
        self.train_step_cnt = agent_params.get("train_step_cnt")
        self.validation_enabled = agent_params.get("validation_enabled")
        self.validation_step_cnt = agent_params.get("validation_step_cnt")
        self.validation_epsilon = agent_params.get("validation_epsilon")

        self.replay_start_size = agent_params.get("replay_start_size")

        self.hidden_size = agent_params.get("hidden_size")
        self.batch_size = agent_params.get("batch_size")
        self.training_freq = agent_params.get("training_freq")
        self.target_model_update_freq = agent_params.get("target_model_update_freq")
        self.gamma = agent_params.get("gamma")
        self.loss_function = agent_params.get("loss_fcn")

        eps_settings = agent_params.get("epsilon")
        self.epsilon_by_frame = self._get_linear_decay_function(
            start=eps_settings["start"],
            end=eps_settings["end"],
            decay=eps_settings["decay"],
            eps_decay_start=self.replay_start_size,
        )

        self._read_and_init_envs()

        buffer_settings = config.get(
            "replay_buffer",
        )
        buffer_type = buffer_settings.get("type")
        if buffer_type is None or buffer_type == "ReplayBuffer":
            self.replay_buffer = ReplayBuffer(
                max_size=buffer_settings.get("max_size"),
                state_dim=self.in_features,
                n_step=buffer_settings.get("n_step"),
            )
        elif buffer_type == "UniqueReplayBuffer":
            self.replay_buffer = UniqueReplayBuffer(
                max_size=buffer_settings.get("max_size"),
                state_dim=self.in_features,
                n_step=buffer_settings.get("n_step"),
            )
        else:
            raise ValueError(
                f"The following buffer type was provided: {buffer_type}, please choose between ReplayBuffer and UniqueReplayBuffer"
            )

        self.logger.info("Loaded configuration settings.")

    def _get_exp_decay_function(self, start: float, end: float, decay: float):
        return lambda x: end + (start - end) * np.exp(-1.0 * x / decay)

    def _get_linear_decay_function(
        self, start: float, end: float, decay: float, eps_decay_start: float = None
    ):
        """Return a function that enables getting the value of epsilon at step x.

        Args:
            start (float): start value of the epsilon function (x=0)
            end (float): end value of the epsilon function (x=decay)
            decay (float): how many steps to reach the end value
            eps_decay_start(float, optional): after how many frames to actually start decaying,
                                            uses self.replay_start_size by default

        Returns:
            function: function to compute the epsillon based on current frame counter
        """
        if not eps_decay_start:
            eps_decay_start = self.replay_start_size

        return lambda x: max(
            end, min(start, start - (start - end) * ((x - eps_decay_start) / decay))
        )

    def _init_models(self, config):
        """Instantiate the policy and target networks.

        Args:
            config (Dict): Settings with parameters for the models

        Raises:
            ValueError: The configuration contains an estimator name that the agent does not
                        know to instantiate.
        """
        env = self.train_env
        states = list(set([s for s, _ in env.mdp.keys()]))
        actions = list(set([a for _, a in env.mdp.keys()]))

        ### Training
        input_size = len(
            states[0]
        )  # Or another way to represent the size of your input
        output_size = len(actions)

        self.policy_model = QNET(input_size, output_size, hidden_size=self.hidden_size)
        self.target_model = QNET(input_size, output_size, hidden_size=self.hidden_size)

        optimizer_settings = config.get("optim")
        self.optimizer = optim.Adam(
            self.policy_model.parameters(), **optimizer_settings["args"]
        )

        self.logger.info("Initialized newtworks and optimizer.")

    def _get_observation_space_shape(self, observation_space):
        """Extract a shape-like tuple from a tuple of discrete spaces."""
        return tuple(space.n for space in observation_space.spaces)

    def _read_and_init_envs(self):
        """Read dimensions of the input and output of the simulation environment"""
        # returns state as [w, h, channels]

        state_shape = self._get_observation_space_shape(
            self.train_env.observation_space
        )

        # permute to get batch, channel, w, h shape
        # specific to minatar
        self.in_features = (1, state_shape[0], state_shape[1])
        self.in_channels = self.in_features[0]
        self.num_actions = self.train_env.action_space.n

        self.train_s = self.train_env.reset()
        self.env_s = self.validation_env.reset()

    def load_models(self, models_load_file):
        checkpoint = torch.load(models_load_file)
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.policy_model.train()
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.target_model.train()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def load_training_stats(self, training_stats_file):
        checkpoint = torch.load(training_stats_file)

        self.t = checkpoint["frame"]
        self.episodes = checkpoint["episode"]
        self.policy_model_update_counter = checkpoint["policy_model_update_counter"]

        self.training_stats = checkpoint["training_stats"]
        self.validation_stats = checkpoint["validation_stats"]

    def save_checkpoint(
        self,
        save_models=True,
        save_training_status=True,
        save_buffer=True,
        save_start_state_counts=True,
    ):
        self.logger.info(f"Saving checkpoint at t = {self.t} ...")
        if save_models:
            self.save_model()
        if save_training_status:
            self.save_training_status()
        if save_buffer:
            self.replay_buffer.save(self.replay_buffer_file)

        if save_start_state_counts:
            start_state_counts_file = os.path.join(
                self.experiment_output_folder,
                f"{self.experiment_name}_start_state_counts.pkl",
            )
            with open(start_state_counts_file, "wb") as f:
                pickle.dump(self.start_state_counts, f)

        self.logger.info(f"Checkpoint saved at t = {self.t}")

    def save_model(self):
        model_file = self._make_model_checkpoint_file_path(
            self.experiment_output_folder, len(self.training_stats)
        )
        Path(os.path.dirname(model_file)).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_model_state_dict": self.policy_model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            model_file,
        )
        self.logger.debug(f"Models saved at t = {self.t}")

    def save_training_status(self):
        status_dict = {
            "frame": self.t,
            "episode": self.episodes,
            "policy_model_update_counter": self.policy_model_update_counter,
            "training_stats": self.training_stats,
            "validation_stats": self.validation_stats,
        }

        torch.save(
            status_dict,
            self.train_stats_file,
        )

        self.logger.debug(f"Training status saved at t = {self.t}")

    def select_action(
        self,
        state: torch.Tensor,
        t: int,
        num_actions: int,
        epsilon: float = None,
        random_action: bool = False,
    ):
        """Select an action with a greedy epsilon strategy.

        Args:
            state (torch.Tensor): The current state
            t (int): the frame number, used in deciding if the selection is done randomly (greedy epsilon)
            num_actions (int): The number of available actions
            epsilon (float, optional): What number to use for the epsilon selection. If None, then will be computed as a function of t.
                                     Defaults to None.
            random_action (bool, optional): Wether to select a random action without making a prediction through the model.
                                         Defaults to False.

        Returns:
            Tuple[int, float]: Returns a tuple. The first element is the selected action (either randomly or by making a model prediction).
                            The second element is the maximum Q value returned by the model. If the random action was selected, the maximum Q is np.Nan.
        """
        max_q = np.nan

        # A uniform random policy
        if random_action:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
            return action

        # Epsilon-greedy behavior policy for action selection
        if not epsilon:
            epsilon = self.epsilon_by_frame(t)

        if np.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
        else:
            action, max_q = self.get_max_q_and_action(state.unsqueeze(0))

        return action, max_q

    def get_max_q_val_for_state(self, state):
        with torch.inference_mode():
            return self.policy_model(state).max(1)[0].item()

    def get_q_val_for_action(self, state, action):
        with torch.inference_mode():
            return torch.index_select(
                self.policy_model(state), 1, action.squeeze(0)
            ).item()

    def get_action_from_model(self, state):
        with torch.inference_mode():
            return self.policy_model(state).max(1)[1].view(1, 1)

    def get_max_q_and_action(self, state):
        with torch.inference_mode():
            maxq_and_action = self.policy_model(state).max(1)
            q_val = maxq_and_action[0].item()
            action = maxq_and_action[1].view(1, 1)
            return action, q_val

    def train(self, train_epochs: int) -> True:
        """The main call for the training loop of the DQN Agent.

        Args:
            train_epochs (int): Represent the number of epochs we want to train for.
                            Note: if the training is resumed, then the number of training epochs that will be done is
                            as many as is needed to reach the train_epochs number.
        """
        if not self.training_stats:
            self.logger.info(f"Starting training session at: {self.t}")
        else:
            epochs_left_to_train = train_epochs - len(self.training_stats)
            self.logger.info(
                f"Resuming training session at: {self.t} ({epochs_left_to_train} epochs left)"
            )
            train_epochs = epochs_left_to_train

        for epoch in range(train_epochs):
            start_time = datetime.datetime.now()

            ep_train_stats = self.train_epoch()
            self.display_training_epoch_info(ep_train_stats)
            self.training_stats.append(ep_train_stats)

            if self.validation_enabled:
                ep_validation_stats = self.validate_epoch()
                self.display_validation_epoch_info(ep_validation_stats)
                self.validation_stats.append(ep_validation_stats)

            if self.save_checkpoints:
                self.save_checkpoint()

            end_time = datetime.datetime.now()
            epoch_time = end_time - start_time

            self.logger.info(
                f"Epoch {epoch} completed in {epoch_time}. Frames seen: {self.t}"
            )
            self.logger.info("\n")

        if self.tensor_board_writer:
            self.tensor_board_writer.close()

        self.logger.info(
            f"Ended training session after {train_epochs} epochs at t = {self.t}"
        )

        return True

    def train_epoch(self) -> Dict:
        """Do a single training epoch.

        Returns:
            Dict: dictionary containing the statistics of the training epoch.
        """
        self.logger.info(f"Starting training epoch at t = {self.t}")
        epoch_t = 0
        policy_trained_times = 0
        target_trained_times = 0

        epoch_episode_rewards = []
        epoch_episode_nr_frames = []
        epoch_losses = []
        epoch_max_qs = []

        start_time = datetime.datetime.now()
        while epoch_t < self.train_step_cnt:
            (
                is_terminated,
                truncated,
                epoch_t,
                current_episode_reward,
                ep_frames,
                ep_policy_trained_times,
                ep_target_trained_times,
                ep_losses,
                ep_max_qs,
            ) = self.train_episode(epoch_t, self.train_step_cnt)

            policy_trained_times += ep_policy_trained_times
            target_trained_times += ep_target_trained_times

            if is_terminated or truncated:
                # we only want to append these stats if the episode ended,
                # otherwise it means it was stopped due to the nr of frames criterion (internal agent tracker)
                epoch_episode_rewards.append(current_episode_reward)
                epoch_episode_nr_frames.append(ep_frames)
                epoch_losses.extend(ep_losses)
                epoch_max_qs.extend(ep_max_qs)

                self.episodes += 1
                self.reset_training_episode_tracker()

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_training_epoch_stats(
            epoch_episode_rewards,
            epoch_episode_nr_frames,
            policy_trained_times,
            target_trained_times,
            epoch_losses,
            epoch_max_qs,
            epoch_time,
        )

        return epoch_stats

    def train_episode(self, epoch_t: int, train_frames: int):
        """Do a single training episode.

        Args:
            epoch_t (int): The total number of frames seen in this epoch, relevant for early stopping of
                            the training episode.
            train_frames (int): How many frames we want to limit the training epoch to

        Returns:
            Tuple[bool, int, float, int, int, int, list, list]: Information relevant to this training episode. Some variables are stored in
                                                            the class so that the training episode can resume in the following epoch.
        """
        policy_trained_times = 0
        target_trained_times = 0

        is_terminated = False
        truncated = False
        while (not is_terminated) and (not truncated) and (
            epoch_t < train_frames
        ):  # can early stop episode if the frame limit was reached
            action, max_q = self.select_action(self.train_s, self.t, self.num_actions)
            action = action.flatten().item()
            s_prime, reward, is_terminated, truncated, info = self.train_env.step(
                action
            )
            s_prime = torch.tensor(s_prime, device=device).float()

            if is_terminated or truncated:
                self.logger.info(
                    f"Episode stopped at t = {self.t} with reward = {reward}. Terminated = {is_terminated}, truncated = {truncated}"
                )

            self.replay_buffer.append(
                self.train_s, action, reward, s_prime, is_terminated
            )

            if self.save_replay_buffer_cycles:
                if self.replay_buffer.did_cycle_occur():
                    replay_buffer_cycles = self.replay_buffer.get_cycle_count()
                    replay_buffer_save_file_name = (
                        f"{self.replay_buffer_file}_{replay_buffer_cycles}"
                    )
                    self.replay_buffer.save(replay_buffer_save_file_name)

            self.max_qs.append(max_q)

            # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
            if (
                self.t > self.replay_start_size
                and len(self.replay_buffer) >= self.batch_size
            ):
                # Train every training_freq number of frames
                if self.t % self.training_freq == 0:
                    if self.normalize_replay_buffer_freq:
                        self.logger.info("Normalizing replay buffer...")
                        normed_replay_buffer = (
                            self.replay_buffer.normalize_replay_buffer()
                        )
                        sample = normed_replay_buffer.sample(self.batch_size)
                    else:
                        sample = self.replay_buffer.sample(self.batch_size)

                    loss_val = self.model_learn(sample)

                    self.losses.append(loss_val)
                    self.policy_model_update_counter += 1
                    policy_trained_times += 1

                # Update the target network only after some number of policy network updates
                if (
                    self.policy_model_update_counter > 0
                    and self.policy_model_update_counter % self.target_model_update_freq
                    == 0
                ):
                    self.target_model.load_state_dict(self.policy_model.state_dict())
                    target_trained_times += 1

            self.current_episode_reward += reward

            self.t += 1
            epoch_t += 1
            self.ep_frames += 1

            # Continue the process
            self.train_s = s_prime

        # end of episode, return episode statistics:
        return (
            is_terminated,
            truncated,
            epoch_t,
            self.current_episode_reward,
            self.ep_frames,
            policy_trained_times,
            target_trained_times,
            self.losses,
            self.max_qs,
        )

    def reset_training_episode_tracker(self):
        """Resets the environment and the variables that keep track of the training episode."""
        self.current_episode_reward = 0.0
        self.ep_frames = 0
        self.losses = []
        self.max_qs = []

        self.train_s = self.train_env.reset()
        self.train_s = torch.tensor(self.train_s, device=device).float()

        hashable_state = tuple(self.train_s.cpu().numpy())
        if hashable_state not in self.start_state_counts:
            self.start_state_counts[hashable_state] = 0
        self.start_state_counts[hashable_state] += 1

    def display_training_epoch_info(self, stats):
        self.logger.info(
            "TRAINING STATS"
            + " | Frames seen: "
            + str(self.t)
            + " | Episode: "
            + str(self.episodes)
            + " | Max reward: "
            + str(stats["episode_rewards"]["max"])
            + " | Avg reward: "
            + str(stats["episode_rewards"]["mean"])
            + " | Avg frames (episode): "
            + str(stats["episode_frames"]["mean"])
            + " | Avg max Q: "
            + str(stats["episode_max_qs"]["mean"])
            + " | Epsilon: "
            + str(self.epsilon_by_frame(self.t))
            + " | Train epoch time: "
            + str(stats["epoch_time"])
        )

    def compute_training_epoch_stats(
        self,
        episode_rewards,
        episode_nr_frames,
        policy_trained_times,
        target_trained_times,
        ep_losses,
        ep_max_qs,
        epoch_time,
    ) -> Dict:
        """Computes the statistics of the current training epoch.

        Args:
            episode_rewards (List): list contraining the final reward of each episode in the current epoch.
            episode_nr_frames (List): list contraining the final number of frames of each episode in the current epoch.
            policy_trained_times (int): Number representing how many times the policy network was updated.
            target_trained_times (int): Number representing how many times the target network was updated.
            ep_losses (List): list contraining losses from the current epoch.
            ep_max_qs (List): list contraining maximum Q values from the current epoch.
            epoch_time (float): How much time the epoch took to compute in seconds.

        Returns:
            Dict: Dictionary with the relevant statistics
        """
        stats = {}

        stats["frame_stamp"] = self.t
        stats["epsilon_greedy"] = self.epsilon_by_frame(self.t)

        stats["episode_rewards"] = self.get_vector_stats(episode_rewards)
        stats["episode_frames"] = self.get_vector_stats(episode_nr_frames)
        stats["episode_losses"] = self.get_vector_stats(ep_losses)
        stats["episode_max_qs"] = self.get_vector_stats(ep_max_qs)

        stats["policy_trained_times"] = policy_trained_times
        stats["target_trained_times"] = target_trained_times
        stats["epoch_time"] = epoch_time

        return stats

    def get_vector_stats(self, vector):
        """Do a single validation epoch.

        Returns:
            Dict: dictionary containing the statistics of the training epoch.
        """
        stats = {}

        if len(vector) > 0:
            stats["min"] = np.nanmin(vector)
            stats["max"] = np.nanmax(vector)
            stats["mean"] = np.nanmean(vector)
            stats["median"] = np.nanmedian(vector)
            stats["std"] = np.nanstd(vector)

        else:
            stats["min"] = None
            stats["max"] = None
            stats["mean"] = None
            stats["median"] = None
            stats["std"] = None

        return stats

    def validate_epoch(self):
        self.logger.info(f"Starting validation epoch at t = {self.t}")

        epoch_episode_rewards = []
        epoch_episode_nr_frames = []
        epoch_max_qs = []
        valiation_t = 0

        start_time = datetime.datetime.now()

        while valiation_t < self.validation_step_cnt:
            (
                current_episode_reward,
                ep_frames,
                ep_max_qs,
            ) = self.validate_episode()

            valiation_t += ep_frames

            epoch_episode_rewards.append(current_episode_reward)
            epoch_episode_nr_frames.append(ep_frames)
            epoch_max_qs.extend(ep_max_qs)

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_validation_epoch_stats(
            epoch_episode_rewards,
            epoch_episode_nr_frames,
            epoch_max_qs,
            epoch_time,
        )
        return epoch_stats

    def compute_validation_epoch_stats(
        self,
        episode_rewards,
        episode_nr_frames,
        ep_max_qs,
        epoch_time,
    ) -> Dict:
        """Computes the statistics of the current validation epoch.

        Args:
            episode_rewards (List): list contraining the final reward of each episode in the current epoch.
            episode_nr_frames (List): list contraining the final number of frames of each episode in the current epoch.
            ep_max_qs (List): list contraining maximum Q values from the current epoch.
            epoch_time (float): How much time the epoch took to compute in seconds.

        Returns:
            Dict: Dictionary with the relevant statistics
        """
        stats = {}

        stats["frame_stamp"] = self.t

        stats["episode_rewards"] = self.get_vector_stats(episode_rewards)
        stats["episode_frames"] = self.get_vector_stats(episode_nr_frames)
        stats["episode_max_qs"] = self.get_vector_stats(ep_max_qs)
        stats["epoch_time"] = epoch_time

        return stats

    def validate_episode(self):
        """Do a single validation episode.

        Returns:
            Tuple[int, int, List, Dict]: Tuple parameters relevant to the validation episode.
                                    The first element is the cumulative reward of the episode.
                                    The second element is the number of frames that were part of the episode.
                                    The third element is a list of the maximum Q values seen.
                                    The fourth element is a dictionary containing the number of times each reward was seen.
        """
        current_episode_reward = 0.0
        ep_frames = 0
        max_qs = []

        # Initialize the environment and start state
        s = self.validation_env.reset()
        s = torch.tensor(s, device=device).float()

        is_terminated = False
        truncated = False
        while (not is_terminated) and (not truncated):  # can early stop episode if the frame limit was reached
            action, max_q = self.select_action(
                s, self.t, self.num_actions, epsilon=self.validation_epsilon
            )
            action = action.flatten().item()
            s_prime, reward, is_terminated, truncated, info = self.validation_env.step(
                action
            )
            s_prime = torch.tensor(s_prime, device=device).float()

            max_qs.append(max_q)

            current_episode_reward += reward

            ep_frames += 1

            # Continue the process
            s = s_prime

        return (current_episode_reward, ep_frames, max_qs)

    def display_validation_epoch_info(self, stats):
        self.logger.info(
            "VALIDATION STATS"
            + " | Max reward: "
            + str(stats["episode_rewards"]["max"])
            + " | Avg reward: "
            + str(stats["episode_rewards"]["mean"])
            + " | Avg frames (episode): "
            + str(stats["episode_frames"]["mean"])
            + " | Avg max Q: "
            + str(stats["episode_max_qs"]["mean"])
            + " | Validation epoch time: "
            + str(stats["epoch_time"])
        )

    def model_learn(self, sample):
        """Compute the loss with TD learning."""
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = sample

        states = torch.stack(states, dim=0)
        next_states = torch.stack(next_states, dim=0)

        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.Tensor(rewards).unsqueeze(1)
        dones = torch.Tensor(dones).unsqueeze(1).type(torch.bool)

        self.policy_model.train()
        self.target_model.eval()

        q_values = self.policy_model(states)
        action_q_values = q_values.gather(1, actions)

        next_q_values = self.target_model(next_states).detach()
        next_q_values = next_q_values.max(1)[0].unsqueeze(1)

        target_q_values = rewards + self.gamma * (next_q_values * (~dones))

        if self.loss_function == "mse_loss":
            loss_fn = nn.MSELoss(reduction="none")
            bellmans_errors = loss_fn(action_q_values, target_q_values)

        if self.loss_function == "smooth_l1":
            loss_fn = nn.SmoothL1Loss(reduction="none")
            bellmans_errors = loss_fn(action_q_values, target_q_values)

        loss = bellmans_errors.mean()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_settings(self):
        """
        Returns the settings of the agent and the environments.

        Returns:
            dict: A dictionary containing the agent settings and environment settings.
        """
        # Extract agent settings dynamically
        agent_settings = vars(self).copy()

        # Remove any unwanted attributes that are not part of the settings
        keys_to_remove = [
            "train_env",
            "validation_env",
            "replay_buffer",
            "logger",
            "tensor_board_writer",
            "config",
        ]
        for key in keys_to_remove:
            agent_settings.pop(key, None)

        # Include epsilon values separately
        agent_settings["epsilon_start"] = self.epsilon_by_frame(0)
        agent_settings["epsilon_end"] = self.epsilon_by_frame(self.train_step_cnt)
        agent_settings["epsilon_decay"] = (
            self.config.get("agent_params", {})
            .get("args", {})
            .get("epsilon", {})
            .get("decay")
        )

        # Extract train environment settings dynamically
        train_env_settings = vars(self.train_env).copy()
        train_env_settings["class"] = self.train_env.__class__.__name__

        # Extract validation environment settings dynamically
        validation_env_settings = vars(self.validation_env).copy()
        validation_env_settings["class"] = self.validation_env.__class__.__name__

        env_settings = {
            "train_env": train_env_settings,
            "validation_env": validation_env_settings,
        }

        # Extract replay buffer settings dynamically
        replay_buffer_settings = vars(self.replay_buffer).copy()
        replay_buffer_settings["class"] = self.replay_buffer.__class__.__name__

        return {
            "agent_settings": agent_settings,
            "env_settings": env_settings,
            "replay_buffer_settings": replay_buffer_settings,
        }


def main():
    pass


if __name__ == "__main__":
    seed_everything(0)
    main()
    # play_game_visual("breakout")
