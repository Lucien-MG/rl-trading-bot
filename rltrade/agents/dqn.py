#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from agents.agent_interface import AgentInterface

from agents.dnn.dqn_linear import SimpleFFDQN
from agents.buffer.replay_buffer import ExperienceReplayBuffer

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent(AgentInterface):
    """ Initialize agent.

    Args:
        config (dict): contains all the parameters needed
    """
    def __init__(self, config):
        self._set_parameters(config)
        
        self.action_space= int(self.action_space)
        self.input_shape = (1, 41)

        self.primary = SimpleFFDQN(self.input_shape, self.action_space).to(self.device)
        self.target = SimpleFFDQN(self.input_shape, self.action_space).to(self.device)
        self.optimizer = optim.SGD(self.primary.parameters(), lr=self.alpha)

        self.primary.apply(self._init_weights)
        self.target.apply(self._init_weights)

        self.memory = ExperienceReplayBuffer(size=self.memory_size)

        self.rewards = deque([0], maxlen=100)

        self.eval_mode = False
        self.step_count = 0
        self.loss = 0

    def _set_parameters(self, configuration):
        self.__dict__ = { k:v for (k,v) in configuration.__dict__.items() }

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            layer.weight.data.uniform_(-0.001, 0.001)
            layer.bias.data.fill_(0.001)

    def _reset_target(self):
        """ Update the target network.
        """
        self.target.load_state_dict(self.primary.state_dict())

    def _update_epsilon(self):
        """ Update the epsilon value.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay_factor, self.min_epsilon)

    @torch.no_grad()
    def _normalize_state(self, state):
        """ Normalize state.
        """
        #mu = torch.mean(state, axis=-1)
        #sigma = torch.std(state, axis=-1)
        #state = torch.transpose((torch.transpose(state, 0, 1) - mu) / (sigma + 1e-10), 0, 1)
        state = (state - torch.mean(state)) / (torch.mean(state) + 1e-10)

        return state

    @torch.no_grad()
    def _normalize_reward(self, reward):
        """ Normalize rewards.
        """
        return reward

    @torch.no_grad()
    def _preprocess_state(self, state):
        """ Apply preprocessing on state.
        """
        state_torch = torch.from_numpy(state)
        # state_torch = torch.unsqueeze(state_torch, axis=-1)

        return state_torch.float()

    def load(self, path):
        """ Load weights at the path location.
        Args:
            path (str): Path where the weights will be loaded.
        """
        self.primary.load_state_dict(torch.load(path))
        self._reset_target()

    def save(self, path):
        """ Save weights at the path location.
        Args:
            path (str): Path where the weights will be saved.
        """
        torch.save(self.primary.state_dict(), path)

    def eval(self):
        """ Turn off exploration and learning.
        """
        self.eval_mode = True

    def train(self):
        """ Turn on exploration and learning.
        """
        self.eval_mode = False

    @torch.no_grad()
    def action(self, state) -> int:
        """ Given the state, select an action.
        Args:
            state (obj): the current state of the environment.

        Returns:
            action (int): an integer compatible with the task's action space.
        """
        state = self._preprocess_state(state)
        state = self._normalize_state(state)

        if not self.eval_mode and torch.rand(1).item() > self.epsilon:
            y_pred = self.primary(state.unsqueeze(dim=0).to(self.device))
        else:
            y_pred = torch.rand(self.action_space)

        action = torch.argmax(y_pred).item()

        return action

    def step(self, state, action, reward, next_state, done) -> None:
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Args:
            state (obj): the previous state of the environment
            action (int): the agent's previous choice of action
            reward (float): last reward received
            next_state (obj): the current state of the environment
            done (bool): whether the episode is complete (True or False)
        """
        # Do not save step or learn in eval mode
        if self.eval_mode:
            return

        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)

        state = self._normalize_state(state)
        next_state = self._normalize_state(next_state)

        reward = self._normalize_reward(reward)

        # Push experience to buffer
        self.memory.push(state, action, reward, next_state, done)

        self.rewards[-1] += reward

        if len(self.memory) >= self.memory_size * 0.2 and self.step_count % self.network_update_step == 0:
            self.learn()

        if self.step_count % self.target_update_step == 0:
            self._reset_target()

        if self.step_count % self.epsilon_update_step == 0:
            self._update_epsilon()

        self.step_count += 1

    def learn(self):
        """ Update the agent's knowledge, using replay buffer.
        """
        # Create random batch of self.batch_size steps
        states, actions, rewards, next_states, dones = self.memory.batch(batch_size=self.batch_size)

        # Create torch tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        # Normalize rewards
        # rewards = self._normalize_reward(rewards)

        # Calculate target reward and detach it from the graph
        # Avoid gradient descend in the target network
        next_state_value = self.target(next_states).max(1)[0].detach()

        # Remove temporal reward if it's the last step
        next_state_value[dones] = 0.0

        # Calculate target reward
        target_reward = rewards + (self.gamma * next_state_value)

        # Actual action values state
        states_action_values = self.primary(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # MSE Error:
        loss = torch.mean(torch.pow(states_action_values - target_reward, 2))

        # Check that the loss is not nan
        self.loss = loss.detach().item()
        assert self.loss == self.loss

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

