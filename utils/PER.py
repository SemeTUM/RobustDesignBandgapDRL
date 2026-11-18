import torch
import random
import numpy as np
# Based on https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
# Adapted for discrete SAC: scalar integer actions, (B, 1) reward/done/weight
# shapes, a configurable device and plain-float priorities in the tree.


# SumTree
class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # leaf index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        # floating-point drift can push cumsum marginally above the stored total
        cumsum = min(cumsum, self.total)

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"


# PER
class PER:
    """Proportional prioritized experience replay.

    Interface mirrors utils.ReplayBuffer where possible:
        add(state, action, reward, next_state, done)
        sample(beta=None) -> (batch, weights, data_idxs)
        update_priorities(data_idxs, td_errors)
        len(buffer)
    batch = (states [B,S], actions [B,1], rewards [B,1], next_states [B,S], dones [B,1])
    weights = importance-sampling weights [B,1], normalised so max == 1.
    """

    def __init__(self, state_size, buffer_size, batch_size, device,
                 eps=1e-2, alpha=0.6, beta=0.4):
        self.tree = SumTree(size=buffer_size)

        self.eps = eps        # keeps every transition sampleable
        self.alpha = alpha    # 0 = uniform sampling, 1 = fully prioritized
        self.beta = beta      # IS-correction strength; anneal towards 1
        self.max_priority = 1.0
        self.batch_size = batch_size
        self.device = device

        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, 1, dtype=torch.float)   # discrete action index
        self.reward = torch.empty(buffer_size, 1, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, 1, dtype=torch.float)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, state, action, reward, next_state, done):
        # new transitions get the current max priority so each is replayed at least once
        self.tree.add(self.max_priority, self.count)

        self.state[self.count] = torch.as_tensor(np.asarray(state), dtype=torch.float).reshape(-1)
        self.action[self.count] = float(np.asarray(action).item())
        self.reward[self.count] = float(reward)
        self.next_state[self.count] = torch.as_tensor(np.asarray(next_state), dtype=torch.float).reshape(-1)
        self.done[self.count] = float(done)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, beta=None):
        batch_size = self.batch_size
        assert self.real_size >= batch_size, "buffer contains fewer transitions than batch_size"
        beta = self.beta if beta is None else beta

        sample_idxs = []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            data_idx, priority, sample_idx = self.tree.get(random.uniform(a, b))
            # guard against landing on an empty / zero-priority leaf
            while sample_idx is None or priority <= 0:
                data_idx, priority, sample_idx = self.tree.get(random.uniform(0, self.tree.total))

            priorities[i] = priority
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -beta
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device),
        )

        return batch, weights.to(self.device), sample_idxs

    def update_priorities(self, data_idxs, td_errors):
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        td_errors = np.abs(np.asarray(td_errors, dtype=np.float64)).reshape(-1)

        for data_idx, err in zip(data_idxs, td_errors):
            priority = float((err + self.eps) ** self.alpha)

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.real_size
