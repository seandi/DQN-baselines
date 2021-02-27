from typing import Optional
import os

import numpy as np
import torch as T

from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer
from logger import Logger


class DQNAgent(object):
    def __init__(
            self,
            gamma,
            lr,
            n_actions,
            input_dims,
            buffer_size,
            batch_size,
            replace=1000,
            algo=None,
            env_name=None,
            models_dir: Optional[str] = None
    ):
        assert models_dir is not None, "No directory where to save the models was given!"

        self.gamma = gamma
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.lr = lr

        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name

        self.models_dir = models_dir

        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.buffer = ReplayBuffer(buffer_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   )

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   )

    def predict_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
        actions = self.q_eval.forward(state)
        action = T.argmax(actions).item()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.store_transition(state, action, reward, next_state, done)

    def sample_buffer(self):
        state, action, reward, new_state, done = \
            self.buffer.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def optimise_agent(self) -> Optional[T.Tensor]:
        if self.buffer.mem_cntr < self.batch_size:
            return None

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_buffer()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        return loss

    def save_models(self, train_steps: int):
        checkpoint_dir = os.path.join(self.models_dir, str(train_steps))
        os.mkdir(checkpoint_dir)
        print("Creating models checkpoint in {0}".format(checkpoint_dir))

        self.q_eval.save_checkpoint(file_name=os.path.join(checkpoint_dir, "q_eval"))
        self.q_next.save_checkpoint(file_name=os.path.join(checkpoint_dir, "q_next"))

    def load_models(self, model_dir):
        self.q_eval.load_checkpoint(os.path.join(model_dir, 'q_eval'))
        self.q_next.load_checkpoint(os.path.join(model_dir, 'q_next'))



