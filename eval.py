import time
from typing import Optional

import numpy as np
from gym import Env

from dqn_agent import DQNAgent
from logger import Logger
from environmentt import make_env


def evaluate(env: Env, agent: DQNAgent, logger: Optional[Logger], epsilon: Optional[float], step: int, eval_episodes: int = 30):
    num_episodes = 0
    eval_rewards = []
    epsilon = epsilon if epsilon is not None else 0.0

    print("Starting evaluation...")
    eval_start_time = time.time()
    while num_episodes < eval_episodes:

        done = False
        episode_reward: float = 0.0
        episode_steps: int = 0
        last_observation = env.reset()

        while not done:
            # 1. Start new interaction
            episode_steps += 1

            # 2. Experience new interaction
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.predict_action(last_observation)
            observation, reward, done, info = env.step(action)

            episode_reward += reward

            last_observation = observation

        num_episodes += 1
        eval_rewards.append(episode_reward)
        print("Episode: {0}. Reward: {1}.".format(num_episodes, episode_reward))

    eval_duration = time.time() - eval_start_time
    max_reward = np.max(eval_rewards)
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)

    if logger is not None:
        logger.log(key='eval\max_reward', value=max_reward, step=step)
        logger.log(key='eval\mean_reward', value=mean_reward, step=step)
        logger.log(key='eval\std_reward', value=std_reward, step=step)
        logger.log(key='eval\duration', value=eval_duration, step=step)
        logger.dump(step=step)
    else:
        print("Eval duration: {0}. Max Reward: {1}. Mean Reward: {2}. Std Reward: {3}. Episodes: {4}".format(
            eval_duration, max_reward, mean_reward, std_reward, eval_episodes
        ))


if __name__ == '__main__':
    log_dir = './runs/DQNAgent-PongNoFrameskip-v4--2021-02-27-15-19'
    models_dir = log_dir + "/models"
    env_name = 'PongNoFrameskip-v4'

    env = make_env(env_name)
    agent = DQNAgent(
        input_dims=(env.observation_space.shape), n_actions=env.action_space.n,
        gamma=0.99, lr=0.0001, batch_size=32, replace=1000, buffer_size=10000,
        models_dir=models_dir, algo='DQNAgent', env_name='PongNoFrameskip-v4'
    )
    agent.load_models(model_dir=models_dir+"/50000")

    evaluate(env=env, agent=agent, logger=None, eval_episodes=10,
             epsilon=0.02, step=50000)
