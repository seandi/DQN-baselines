from typing import Optional, List, Tuple

import numpy as np
from gym import Env

from DQN.dqn_agent import DQNAgent
from DQN.replay_memory import ReplayBuffer


class Trainer:
    def __init__(
            self,
            env: Env,
            agent: DQNAgent,
            replay_buffer: ReplayBuffer,
            max_interaction_steps: int = 1e6,
            max_episodes: Optional[int] = None,
            epsilon_start: float = 1.0,
            epsilon_min: float = 0.1,
            epsilon_decay_period: Optional[int] = int(2e5),
            epsilon_decay_factor: Optional[int] = None


    ):
        assert epsilon_decay_period is not None or epsilon_decay_factor is not None, "Epsilon schedule not specified!"

        self.env = env
        self.agent: DQNAgent = agent
        self.replay_buffer = replay_buffer

        self.max_steps: float = float(max_interaction_steps) if max_interaction_steps is not None else float('inf')
        self.max_episodes: float = float(max_episodes) if max_episodes is not None else float('inf')


        self.episode: int = 0
        self.interaction_steps: int = 0
        self.train_steps: int = 0
        self.train_history: List[Tuple[int, float]] = []

        self.epsilon: float = 0
        self.epsilon_start: float = epsilon_start
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay_period: int = epsilon_decay_period  # Linear decay over this interaction steps
        self.epsilon_decay_factor: int = epsilon_decay_factor  # Exponential decay

    def update_epsilon(self):
        if self.interaction_steps == 0:
            self.epsilon = self.epsilon_start
            self.epsilon_step = (self.epsilon_start - self.epsilon_min) / self.epsilon_decay_period
        else:
            self.epsilon -= self.epsilon_step

    def select_action(self, observation):
        if np.random.rand() < self.epsilon:
            # Choose randomly
            action = self.env.action_space.sample()
        else:
            # Select greedily using the agent policy net
            action = self.agent.predict_action(observation=observation)

        return action

    def train(self) -> List[Tuple[int, float]]:
        self.episode = 0
        episode_reward: float = 0.0
        episode_steps: int = 0
        last_observation = self.env.reset()
        # Setup epsilon schedule
        self.update_epsilon()

        while self.interaction_steps < self.max_steps and self.episode < self.max_episodes:
            # 1. Start new interaction
            self.interaction_steps += 1
            episode_steps += 1

            # 2. Experience new interaction and add to buffer
            action = self.select_action(last_observation)
            observation, reward, done, info = self.env.step(action)

            episode_reward += reward

            self.replay_buffer.store_transition(
                state=last_observation,
                action=action,
                reward=reward,
                next_state=observation,
                done=done
            )

            # 3. Optimise agent
            self.train_steps += 1
            self.agent.optimise_agent()

            #4. Complete interaction step
            self.update_epsilon()
            if done:
                self.train_history.append((episode_steps, episode_reward))
                # start new episode
                episode_reward = 0.0
                last_observation = self.env.reset()
                self.episode += 1


                print("Episode: {0}. Reward: {1}. Steps: {2}. Total_steps: {3}. Epsilon: {4:.2f}".format(
                    self.episode, self.train_history[-1][1], self.train_history[-1][0],
                    self.interaction_steps, self.epsilon)
                )

            else:
                last_observation = observation

        return self.train_history














