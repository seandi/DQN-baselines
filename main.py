from dqn_agent import DQNAgent
from trainer import Trainer
from environmentt import make_env
from utils import make_dirs
from logger import Logger
from epsilon_scheduler import EpsilonScheduler


if __name__ == '__main__':
    env_name = 'PongNoFrameskip-v4'

    env = make_env(env_name)

    run_name = DQNAgent.__name__ + "-" + env_name + "-"
    log_dir, models_dir = make_dirs('runs', run_name, add_run_time=True)

    agent = DQNAgent(
        input_dims=(env.observation_space.shape), n_actions=env.action_space.n,
        gamma=0.99, lr=0.0001, batch_size=32, replace=1000, buffer_size=10000,
        models_dir=models_dir, algo='DQNAgent', env_name='PongNoFrameskip-v4'
    )

    epsilon_scheduler = EpsilonScheduler(epsilon_start=1.0, epsilon_min=0.02,
                                         epsilon_decay_factor=0.999985, schedule='exponential'
                                         )
    replay_buffer = agent.buffer

    logger = Logger(log_dir=log_dir, use_tensorboard=True)

    trainer = Trainer(
        env=env, agent=agent, replay_buffer=replay_buffer, logger=logger,
        epsilon_scheduler=epsilon_scheduler,
        max_interaction_steps=1000000, save_models_interval=100000,
        eval_interval=None, eval_episodes=30, eval_epsilon=0.02
    )

    trainer.train()
