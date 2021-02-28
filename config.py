from configparser import ConfigParser
from typing import Optional
import dataclasses

DASHED_LINE = "-------------------------------------------------------------\n"
SUMMARY_HEADER = DASHED_LINE + \
                 "            HYPER-PARAMETERS CONFIGURATION                   \n" \
                 + DASHED_LINE

@dataclasses.dataclass
class ConfigDict:
    # ---------- GENERIC HYPER-PARAMS ----------- #
    device_name: str
    disable_cuda: bool

    # ---------- TRAINING HYPER-PARAMS ----------- #
    interaction_steps: int
    max_episodes: Optional[int]
    train_agent_frequency: int
    save_model_interval: int
    epsilon_start: float
    epsilon_min: float
    epsilon_scheduler: str
    epsilon_decay_factor: Optional[float]
    epsilon_decay_period: Optional[int]
    eval_interval: Optional[int]
    eval_episodes: int
    eval_epsilon: float
    start_optimizing_agent_after: int

    # ---------- AGENT HYPER-PARAMS ----------- #
    learning_rate: float
    batch_size: int
    gamma: float
    buffer_size: int
    optimizer: str
    update_target_net_params: int

    def __init__(self, config_file: str) -> None:
        parser: ConfigParser = ConfigParser()
        parser.read(config_file)

        # ---------- GENERIC HYPER-PARAMS ----------- #
        self.disable_cuda = parser.getboolean('GENERAL', 'disable_cuda')

        # ---------- TRAINING HYPER-PARAMS ----------- #
        self.interaction_steps = int(parser.get('TRAINING', 'interaction_steps'))

        t: str = parser.get('TRAINING', 'max_episodes')
        self.max_episodes = int(t) if t != 'None' else None

        self.train_agent_frequency = int(parser.get('TRAINING', 'train_agent_frequency'))
        self.save_model_interval = int(parser.get('TRAINING', 'save_model_interval'))
        self.epsilon_start = float(parser.get('TRAINING', 'epsilon_start'))
        self.epsilon_min = float(parser.get('TRAINING', 'epsilon_min'))
        self.epsilon_scheduler = str(parser.get('TRAINING', 'epsilon_scheduler'))

        t: str = (parser.get('TRAINING', 'epsilon_decay_factor'))
        self.epsilon_decay_factor = float(t) if t != "None" else None

        t: str = (parser.get('TRAINING', 'epsilon_decay_period'))
        self.epsilon_decay_period = int(t) if t != 'None' else None

        t: str = (parser.get('TRAINING', 'eval_interval'))
        self.eval_interval = int(t) if t != 'None' else None

        self.eval_episodes = int(parser.get('TRAINING', 'eval_episodes'))
        self.eval_epsilon = float(parser.get('TRAINING', 'eval_epsilon'))
        self.start_optimizing_agent_after = int(parser.get('TRAINING', 'start_optimizing_agent_after'))

        # ---------- AGENT HYPER-PARAMS ----------- #
        self.learning_rate = float(parser.get('AGENT', 'learning_rate'))
        self.batch_size = int(parser.get('AGENT', 'batch_size'))
        self.buffer_size = int(parser.get('AGENT', 'buffer_size'))
        self.gamma = float(parser.get('AGENT', 'gamma'))
        self.update_target_net_params = int(parser.get('AGENT', 'update_target_net_params'))
        self.optimizer = str(parser.get('AGENT', 'optimizer'))

    def generate_summary(self, save_to_file: Optional[str]=None) -> str:
        config_dict = self.__dict__
        longest_param_name = max([len(param) for param in config_dict.keys()])

        summary_lines = []
        for param in config_dict:
            summary_lines.append("{0:<{2}}: {1}\n".format(param, config_dict[param], longest_param_name + 2))
        summary_string = SUMMARY_HEADER + "".join(summary_lines) + DASHED_LINE

        if save_to_file is not None:
            with open(save_to_file, 'w') as file:
                file.write(summary_string)
                file.flush()

        return summary_string


if __name__ == '__main__':
    c = ConfigDict(config_file='configs/config_atari_dqn.ini')
    summary = c.generate_summary(save_to_file='./parameters.txt')
    print(summary)
