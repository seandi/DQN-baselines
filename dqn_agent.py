from typing import Optional
import os

import numpy as np
import torch as T
from gym import Env

from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer
from environmentt import get_num_action, get_observation_shape


class DQNAgent(object):
    def __init__(
            self,
            device: T.device,
            gamma: float,
            learning_rate: float,
            env: Env,
            buffer_size: int,
            batch_size: int,
            # Measured in TRAIN steps, hyper-param are usually wrt INTERACTION steps
            sync_target_net_every_n_train_steps: int,
            models_dir: Optional[str] = None,
    ):
        assert models_dir is not None, "No directory where to save the models was given!"

        # 1. Store hyper-parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = learning_rate
        self.target_net_update_freq = sync_target_net_every_n_train_steps
        self.models_dir = models_dir
        self.device = device

        # 2. Setup agent
        self.train_step = 0

        input_shape = get_observation_shape(env.observation_space)
        num_actions = get_num_action(env.action_space)

        self.buffer = ReplayBuffer(buffer_size, input_shape=input_shape, n_actions=num_actions)

        self.policy_net = DeepQNetwork(input_dims=input_shape, n_actions=num_actions).to(self.device)
        self.target_net = DeepQNetwork(input_dims=input_shape, n_actions=num_actions).to(self.device)
        self.update_target_network()

        self.loss = T.nn.MSELoss()
        self.optimizer = T.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def predict_action(self, observation) -> int:
        state = T.tensor([observation], dtype=T.float).to(self.device)
        actions = self.policy_net.forward(state)
        action = int(T.argmax(actions).item())

        return action

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.buffer.store_transition(state, action, reward, next_state, done)

    def sample_buffer(self):
        state, action, reward, new_state, done = \
            self.buffer.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.device)
        rewards = T.tensor(reward).to(self.device)
        dones = T.tensor(done).to(self.device)
        actions = T.tensor(action).to(self.device)
        states_ = T.tensor(new_state).to(self.device)

        return states, actions, rewards, states_, dones

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimise_agent(self) -> Optional[T.Tensor]:
        if self.buffer.mem_cntr < self.batch_size:
            return None

        self.optimizer.zero_grad()

        observations, actions, rewards, next_observations, dones = self.sample_buffer()

        indices = np.arange(self.batch_size)
        policy_q_values = self.policy_net.forward(observations)[indices, actions]

        with T.no_grad():
            q_next = self.target_net.forward(next_observations).max(dim=1)[0]
            q_next[dones] = 0.0
            target_q_values = rewards + self.gamma * q_next

        loss_t = self.loss(target_q_values, policy_q_values)
        loss_t.backward()
        self.optimizer.step()
        self.train_step += 1

        if self.train_step % self.target_net_update_freq == 0:
            self.update_target_network()

        return loss_t

    def save_models(self, train_steps: int):
        checkpoint_dir = os.path.join(self.models_dir, str(train_steps))
        os.mkdir(checkpoint_dir)
        print("Creating models checkpoint in {0}".format(checkpoint_dir))

        self.policy_net.save_checkpoint(file_name=os.path.join(checkpoint_dir, "q_eval"))
        self.target_net.save_checkpoint(file_name=os.path.join(checkpoint_dir, "q_next"))

    def load_models(self, model_dir):
        self.policy_net.load_checkpoint(os.path.join(model_dir, 'q_eval'))
        self.target_net.load_checkpoint(os.path.join(model_dir, 'q_next'))



