import torch
import torch.nn as nn
import numpy as np
import random

# essentially copied from ./unia with some modifications for memory (don't use done)

class Memory:
    def __init__(self, mem_size: int, batch_size: int):
        self.mem = []
        self.mem_size = mem_size
        self.batch_size = batch_size

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, nstate: torch.Tensor):
        self.mem.append((state, action, reward, nstate))
        if len(self.mem) > self.mem_size:
            self.mem.pop(0)

    def random_sample(self):
        if len(self.mem) < self.batch_size:
            return False
        return random.sample(self.mem, self.batch_size)

class odN(nn.Module): # small net
    def __init__(self, state_shape: int, action_shape: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_shape, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, action_shape)
        )

    def forward(self, x_in: torch.Tensor):
        return self.net(x_in)

class Agent(nn.Module):
    def __init__(self, obs_shape: torch.Tensor | tuple | int, action_shape: torch.Tensor | int,
                 mem_size: int = 32768, batch_size: int = 32, gamma: float = 0.999, eps_start: float = 1,
                 eps_end: float = 0.05, steps: int = 2*10**2, optimizer: torch.optim.Optimizer | None = None, learning_rate: float = 0.0001,
                 loss: nn.modules.loss.Module | None = None, device: str = "cpu"): # mem_size: choose large size, but not too large, such that it would use swap

        super().__init__()

        self.obs_shape = obs_shape
        self.action_space = action_shape

        self.main_network = odN(obs_shape, action_shape)
        self.target_network = odN(obs_shape, action_shape)
        self.replay_buffer = Memory(mem_size, batch_size)

        self.batch_size = batch_size
        self.gamma = gamma

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.steps = steps
        self.step = 0
        self.decay = eps_start

        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate) if optimizer is None else optimizer
        self.lr = learning_rate
        self.loss = loss if loss is not None else nn.SmoothL1Loss()

        self.device = torch.device(device)

        self.main_network = self.main_network.to(self.device)
        self.target_network = self.target_network.to(self.device)

    def select_action(self, observations: torch.Tensor) -> int:
        self.update_decay()
        eps = random.random()

        if eps < self.decay:
            return random.randint(0, self.action_space-1)

        self.main_network.eval()
        with torch.no_grad():
            return int(torch.argmax(self.main_network(observations.to(self.device)), -1))

    def update_decay(self):
        self.decay = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.step / self.steps)
        self.step += 1

    def training_main(self) -> float:
        self.main_network.train()
        self.target_network.eval()

        self.optim.zero_grad(True)

        sample = self.replay_buffer.random_sample()

        if isinstance(sample, bool):
            return 0.

        state_ten, action_ten, reward_ten, nstate_ten = zip(*sample)

        state_ten = torch.stack(state_ten).to(self.device)
        action_ten = torch.stack(action_ten).to(self.device, dtype=torch.int64)
        rew_ten = torch.stack(reward_ten).to(self.device)
        nstate_ten = torch.stack(nstate_ten).to(self.device)

        out = self.main_network(state_ten).gather(dim=1, index=action_ten)

        with torch.no_grad():
            q_next_state = self.target_network(nstate_ten)

        target = rew_ten + self.gamma*torch.max(q_next_state, dim=-1).values.unsqueeze(1)

        loss = self.loss(out, target)
        loss.backward()

        self.optim.step()

        return loss.item()