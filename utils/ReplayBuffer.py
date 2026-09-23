import random
import math
from collections import namedtuple, deque
import torch.autograd as grad
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import copy

#-----REPLAY BUFFER
from collections import deque, namedtuple

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class ReplayBufferBias:
    def __init__(self, buffer_size, batch_size, device):
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, positive_fraction=0.8):
        # Separate positive and negative rewards
        positive_exps = [e for e in self.memory if e.reward > 0]
        negative_exps = [e for e in self.memory if e.reward <= 0]

        # Determine how many of each to sample
        n_pos = int(self.batch_size * positive_fraction)
        n_neg = self.batch_size - n_pos

        pos_sample = random.choices(positive_exps, k=n_pos) if len(positive_exps) > 0 else []
        neg_sample = random.choices(negative_exps, k=n_neg) if len(negative_exps) > 0 else []

        experiences = pos_sample + neg_sample
        random.shuffle(experiences)

        states = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
