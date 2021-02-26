from dqn_agent import DQNAgent
from trainer import Trainer
from utils import make_env


if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')

    agent = DQNAgent(
        input_dims=(env.observation_space.shape), n_actions=env.action_space.n,
        gamma=0.99, lr=0.0001, batch_size=32, replace=1000, buffer_size=10000,
        chkpt_dir='models/', algo='DQNAgent', env_name='PongNoFrameskip-v4'
    )

    replay_buffer = agent.buffer

    trainer = Trainer(
        env=env, agent=agent, replay_buffer=replay_buffer,
        max_interaction_steps=1000000,
        epsilon_start=1.0, epsilon_min=0.1, epsilon_decay_period=200000
    )

    trainer.train()