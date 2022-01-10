import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.gamma = gamma

        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)

        # Handle training step with only 1 iteration aka shape: (1, x)
        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            next_states = torch.unsqueeze(next_states, 0)
            dones = (dones, )

        # 1: Predicted Q values with current state
        pred = self.model(states)

        # 2: reward + gamma * max(next predicted Q value) (only if not done)
        target = pred.clone()

        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + rewards[idx] + self.gamma * \
                    torch.max(self.model(next_states[idx]))

            target[idx][torch.argmax(actions[idx]).item()] = Q_new

        # Empty gradient
        self.optimizer.zero_grad()
        # calculate loss
        loss = self.criterion(target, pred)
        # Backpropagate
        loss.backward()

        self.optimizer.step()
