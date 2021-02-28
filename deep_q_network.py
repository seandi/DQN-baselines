
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, net_arch):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self, file_name: str):
        T.save(self.state_dict(), file_name)

    def load_checkpoint(self, file_name: str):
        self.load_state_dict(T.load(file_name))


class MLP_DQN(nn.Module):
    def __init__(self, input_dims, n_actions, net_arch=(128, 128)):
        super(MLP_DQN, self).__init__()

        net = nn.ModuleList()
        net.extend([
            nn.Linear(input_dims[0], net_arch[0]),
            nn.ReLU()
        ])
        for layer in range(len(net_arch)-1):
            net.extend([
                nn.Linear(net_arch[layer], net_arch[layer+1]),
                nn.ReLU()
            ])

        net.extend([
            nn.Linear(net_arch[-1], n_actions),
            nn.ReLU()
        ])
        self.model = nn.Sequential(*net)

    def forward(self, state):
        return  self.model(state)

    def save_checkpoint(self, file_name: str):
        T.save(self.state_dict(), file_name)

    def load_checkpoint(self, file_name: str):
        self.load_state_dict(T.load(file_name))
