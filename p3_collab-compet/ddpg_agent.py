import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 3e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 2e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

NUM_AGENTS = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size, action_size).to(device)
        self.actor_target = Actor(
            state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            state_size * NUM_AGENTS, action_size * NUM_AGENTS).to(device)
        self.critic_target = Critic(
            state_size * NUM_AGENTS, action_size * NUM_AGENTS).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noises = [OUNoise(action_size, random_seed) for _ in range(NUM_AGENTS)]

        # Replay memory
        self.memory = ReplayBuffer(
            action_size, BUFFER_SIZE, BATCH_SIZE)

        # Make sure local and target networks are in sync
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy()
        self.actor_local.train()
        if add_noise:
            action += [n.sample() for n in self.noises]
        return np.clip(action, -1, 1)

    def reset(self):
        for noise in self.noises:
            noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Sample experiences and convert to tensor
        experiences = list(map(lambda x:torch.tensor(x, dtype=torch.float).to(device), zip(*experiences)))
        
        states, states_vec, actions, rewards, next_states, next_states_vec, dones = experiences

        # Compute actions for target actor network
        with torch.no_grad():
            target_next_actions = [self.actor_target(
                next_states[:, i, :]) for i in range(NUM_AGENTS)]

        target_next_actions = torch.cat(target_next_actions, dim=1)

        # Compute Q values for the next states and next actions with the target critic model
        with torch.no_grad():
            next_Q_value_target = self.critic_target(
                next_states_vec, target_next_actions)

        # Compute Q values for the current states and actions
        Q_value_target = rewards.sum(1, keepdim=True) + GAMMA * next_Q_value_target * (1 - dones.max(1, keepdim=True)[0])

        # Compute Q values for the current states and actions with the local critic model
        actions = actions.view(actions.shape[0], -1)
        Q_value_local = self.critic_local(states_vec, actions)

        # Compute and minimize the local critic loss
        critic_local_loss = F.mse_loss(Q_value_local, Q_value_target.detach())
        self.critic_optimizer.zero_grad()
        critic_local_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Compute actions for the current states and actions for the actor network
        actions_local = [self.actor_local(
            states[:, i, :]) for i in range(NUM_AGENTS)]
        actions_local = torch.cat(actions_local, dim=1)

        # Compute and minimize the local actor loss
        actor_local_loss = -self.critic_local(states_vec, actions_local).mean()
        self.actor_optimizer.zero_grad()
        actor_local_loss.backward()
        self.actor_optimizer.step()

        # Soft update target critic and actor models
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "states", "states_vector", "actions", "rewards", "next_states", "next_states_vector", "dones"])

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory for multiple agents"""
        e = self.experience(states, states.flatten(), actions, rewards, next_states, next_states.flatten(), dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
