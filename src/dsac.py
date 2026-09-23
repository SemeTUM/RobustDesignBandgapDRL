#write sac discrete agent function to train only based on the design
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import os
import random
import math
import numpy as np
from collections import namedtuple, deque
import torch.autograd as grad
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import copy

def hidden_init(layer):
    fan_in = layer.weight.data.size()[1]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
 
 
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=80):
        super(Actor, self).__init__()
 
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

        self.reset_parameters()
 
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-4e-3, 4e-3)
 
    def forward(self, state):
 
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_probs = self.softmax(self.fc4(x))
        return action_probs
 
    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)
 
        dist = Categorical(action_probs)
        action = dist.sample()
 
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
 
    def get_action(self, state):
        """Stochastic action sample from the categorical policy."""
        action_probs = self.forward(state)
 
        dist = Categorical(action_probs)
        action = dist.sample()
 
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
 
    def get_det_action(self, state):
        action_probs = self.forward(state)
        action = torch.argmax(action_probs, dim=-1)
        return action.detach().cpu()
 
 
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=80, seed=1):
        super(Critic, self).__init__()

        self.seed = seed
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()
 
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-4e-3, 4e-3)
 
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


####
class SAC(nn.Module):
    
    def __init__(self,  state_size, action_size, device):
        super(SAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size


        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = 80
        learning_rate = 6e-4
        self.clip_grad_param = 1
        self.device= device

        self.target_entropy = 0.98 * (-np.log(1.0 / action_size))  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
                
        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(self.device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(self.device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 

    
    def get_action(self, state, deterministic=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor_local.get_det_action(state)
            else:
                action, _, _ = self.actor_local.get_action(state)
        return action.numpy()


    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        with torch.no_grad():
            q1 = self.critic1(states)
            q2 = self.critic2(states)
            min_Q = torch.min(q1, q2)
 
        actor_loss = (action_probs * (alpha * log_pis - min_Q)).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi

    
    def learn(self, step, experiences, gamma, d=1):
        states, actions, rewards, next_states, dones = experiences
 
        # update critics 
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha * log_pis)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1))
 
        # Compute critic loss
        q1 = self.critic1(states).gather(1, actions.long())
        q2 = self.critic2(states).gather(1, actions.long())
 
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)
 
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        # clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        # clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()
 
        # update actor against the freshly updated critics ----
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
 
        # ---- Compute alpha loss ----
        # gradient by alpha itself and stalls once alpha gets small.
        alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()
 
        # update target networks
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
 
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

    @torch.no_grad()
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
