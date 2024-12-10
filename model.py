import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using torch device:", device)
print(torch.cuda.get_device_name())


class Linear_QNet(nn.Module):
    def __init__(self, topology) -> None:
        super().__init__()

        self.topology = topology
        self.layers = nn.ModuleList().to(device)

        # Input, hidden and output layers
        for i, layer in enumerate(topology[:-1]):
            next_layer = topology[i + 1]
            self.layers.append(nn.Linear(layer, next_layer).to(device))

    def __str__(self) -> str:
        return f"Linear_QNet(" + ",".join([str(x) for x in self.topology]) + ")"

    def forward(self, x):
        # Input
        x = F.relu(self.layers[0](x)).to(device)
        # Hidden
        for hidden_layer in self.layers[1:-1]:
            x = F.relu(hidden_layer(x)).to(device)
        # Out
        return self.layers[-1](x)

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            raise(ValueError("Could not find model path"))

        file_name = os.path.join(model_folder_path, file_name)

        self.load_state_dict(torch.load(file_name))
        self.eval().to(device)


class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.gamma = gamma

        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss().to(device)

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(device)
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float).to(device)

        # Handle training step with only 1 iteration aka shape: (1, x)
        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0).to(device)
            actions = torch.unsqueeze(actions, 0).to(device)
            rewards = torch.unsqueeze(rewards, 0).to(device)
            next_states = torch.unsqueeze(next_states, 0).to(device)
            dones = (dones, )

        # 1: Predicted Q values with current state
        pred = self.model(states)

        # 2: reward + gamma * max(next predicted Q value) (only if not done)
        target = pred.clone()

        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * \
                    torch.max(self.model(next_states[idx])).to(device)

            target[idx][torch.argmax(actions[idx]).to(device).item()] = Q_new

        # Empty gradient
        self.optimizer.zero_grad()
        # calculate loss
        loss = self.criterion(target, pred)
        # Backpropagate
        loss.backward()
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
