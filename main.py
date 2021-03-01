import torch as th
import gym
import os

from dqn_agent import DQNAgent
from trainer import Trainer
from environmentt import make_atari_env
from utils import make_dirs
from logger import Logger
from epsilon_scheduler import EpsilonScheduler
from config import ConfigDict

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', default='PongNoFrameskip-v4',
                        help="Name of the gym environment to solve")
    parser.add_argument('--config', default='./configs/config_atari_dqn.ini',
                        help="Config file containing the hyper-params")
    parser.add_argument('--atari', default=True, help="Is the env an Atari env")
    args = parser.parse_args()

    env_name = args.env
    config_filename = args.config

    config = ConfigDict(config_file=config_filename)

    device = th.device("cuda" if th.cuda.is_available() and not config.disable_cuda else "cpu")
    device_name: str = th.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
    config.device_name = device_name

    env = make_atari_env(env_name) if args.atari else gym.make(env_name)

    run_name = DQNAgent.__name__ + "-" + env_name + "-"
    log_dir, models_dir = make_dirs('runs', run_name, add_run_time=True)

    summary = config.generate_summary(save_to_file=os.path.join(log_dir, 'params.txt'))
    print(summary)

    agent = DQNAgent(
        device=device, env=env,
        gamma=config.gamma, learning_rate=config.learning_rate, batch_size=config.batch_size,
        sync_target_net_every_n_train_steps=config.update_target_net_params//config.train_agent_frequency,
        buffer_size=config.buffer_size,
        models_dir=models_dir, model=config.model, net_arch=config.net_arch
    )

    epsilon_scheduler = EpsilonScheduler(epsilon_start=config.epsilon_start, epsilon_min=config.epsilon_min,
                                         epsilon_decay_factor=config.epsilon_decay_factor,
                                         epsilon_decay_period=config.epsilon_decay_period,
                                         schedule=config.epsilon_scheduler
                                         )
    replay_buffer = agent.buffer

    logger = Logger(log_dir=log_dir, use_tensorboard=True)

    trainer = Trainer(
        env=env, agent=agent, replay_buffer=replay_buffer, logger=logger,
        epsilon_scheduler=epsilon_scheduler,
        max_interaction_steps=config.interaction_steps, max_episodes=config.max_episodes,
        save_models_interval=config.save_model_interval, train_freq=config.train_agent_frequency,
        train_start=config.start_optimizing_agent_after,
        eval_interval=None, eval_episodes=config.eval_episodes, eval_epsilon=config.eval_epsilon
    )

    trainer.train()
