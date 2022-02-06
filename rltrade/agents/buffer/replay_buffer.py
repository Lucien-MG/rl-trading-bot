#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from random import choices
from collections import deque

class ExperienceReplayBuffer():
    """ Initialize ExperienceReplayBuffer.

    Args:
        size (int): Replay buffer's size.
    """
    def __init__(self, size=100):
        self.buffer = deque(maxlen=size)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def batch(self, batch_size=32):
        steps = choices(self.buffer, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*steps)
        return states, actions, rewards, next_states, dones
