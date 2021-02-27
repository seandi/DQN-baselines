from typing import Optional, List, Tuple
import time

import numpy as np
from gym import Env

from dqn_agent import DQNAgent
from replay_memory import ReplayBuffer
from logger import Logger
from epsilon_scheduler import EpsilonScheduler
from eval import evaluate


class Trainer:

    def __init__(
            self,
            env: Env,
            agent: DQNAgent,
            replay_buffer: ReplayBuffer,
            epsilon_scheduler: EpsilonScheduler,
            logger: Logger,
            max_interaction_steps: int = 1e6,
            max_episodes: Optional[int] = None,
            save_models_interval: Optional[int] = None,  # measured in train steps
            eval_interval: Optional[int] = None,
            eval_episodes: int = 30,
            eval_epsilon: Optional[float] = None
    ):
        self.env = env
        self.agent: DQNAgent = agent
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.eps_scheduler = epsilon_scheduler

        self.max_steps: float = float(max_interaction_steps) if max_interaction_steps is not None else float('inf')
        self.max_episodes: float = float(max_episodes) if max_episodes is not None else float('inf')
        self.model_checkpoint_freq = save_models_interval if save_models_interval is not None else int(1e6)

        # Evaluate every this number of interactions (after current episode ends)
        self.eval_interval = eval_interval if eval_interval is not None else int(1e9)
        self.eval_episodes = eval_episodes
        self.eval_epsilon = eval_epsilon if eval_epsilon is not None else self.eps_scheduler.epsilon_min

        self.episode: int = 0
        self.interaction_steps: int = 0
        self.train_steps: int = 0
        self.train_history: List[Tuple[int, float]] = []

    def select_action(self, observation):
        epsilon = self.eps_scheduler.step()
        self.logger.log(key='train/epsilon', value=epsilon, step=self.interaction_steps)
        if np.random.rand() < epsilon:
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
        eval_on_episode_end = False

        episode_start_time = time.time()
        while self.interaction_steps < self.max_steps and self.episode < self.max_episodes:
            # 1. Start new interaction
            self.interaction_steps += 1
            episode_steps += 1
            if self.interaction_steps % self.eval_interval == 0:
                eval_on_episode_end = True

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
            loss = self.agent.optimise_agent()
            if loss is not None:
                self.logger.log(key='train/policy_net_loss', value=loss, step=self.interaction_steps)

            if self.train_steps % self.model_checkpoint_freq == 0:
                self.agent.save_models(self.train_steps)

            # 4. Complete interaction step
            if done:
                self.episode += 1
                duration = time.time() - episode_start_time
                self.train_history.append((episode_steps, episode_reward))

                self.logger.log(key='train/episode', value=self.episode, step=self.interaction_steps)
                self.logger.log(key='train/episode_reward', value=episode_reward, step=self.interaction_steps)
                self.logger.log(key='train/duration', value=duration, step=self.interaction_steps)

                history_len = int(min(100, len(self.train_history)))
                reward_mov_avg = [episode[1] for episode in self.train_history[-history_len:]]
                reward_mov_avg = float(np.mean(reward_mov_avg))
                self.logger.log(key='train/reward_avg_last_100', value=reward_mov_avg, step=self.interaction_steps)

                self.logger.dump(step=self.interaction_steps)

                if eval_on_episode_end:
                    evaluate(env=self.env, agent=self.agent, logger=self.logger,
                             step=self.interaction_steps, eval_episodes=self.eval_episodes,
                             epsilon=self.eval_epsilon
                             )
                    eval_on_episode_end = False

                # start new episode
                episode_reward = 0.0
                last_observation = self.env.reset()
                episode_start_time = time.time()

            else:
                last_observation = observation

        self.agent.save_models(train_steps=self.train_steps + 1)

        evaluate(env=self.env, agent=self.agent, logger=self.logger,
                 step=self.interaction_steps, eval_episodes=self.eval_episodes,
                 epsilon=self.eval_epsilon
                 )

        return self.train_history



