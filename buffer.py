from collections import namedtuple, deque
import random
from utilities import transpose_list
import torch
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.FloatTensor([e.state for e in experiences if e is not None]).to(self.device)
        actions = torch.FloatTensor([e.action for e in experiences if e is not None]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences if e is not None]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences if e is not None]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences if e is not None]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
