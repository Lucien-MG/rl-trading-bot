#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

from .csvlog import CsvLogger

class RunEnv:
    """ Initialize the RunEnv class.
    Args:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
    Attributes:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
    """
    def __init__(self, env, agent, log_path=None, limit_step=None, render=None):
        self.env = env
        self.agent = agent

        self.limit_step = limit_step
        self.render = render

        self.rewards = []

        self.logger = CsvLogger(log_path, ["reward", "cash", "stock", "done"]) if log_path else None

    def step(self, state):
        # agent choose action with state observation
        action = self.agent.action(state)

        # agent performs the selected action
        next_state, reward, done, info = self.env.step(action)

        # agent performs internal updates based on sampled experience
        self.agent.step(state, action, reward, next_state, done)

        return next_state, reward, done, info

    def episode(self):
        # begin the episode
        state = self.env.reset()
        step = 0

        if self.logger:
            self.logger.open()

        while True:
            # Use the render mode if needed
            if self.render:
                self.env.render()

            # Execute one step: the agent take an action and the env return the new state
            next_state, reward, done, info = self.step(state)

            # update the sampled reward
            self.rewards.append(reward)

            # update the state (s <- s') to next time step
            state = next_state

            step += 1

            # Write logs:
            if self.logger:
                self.logger.register(
                    {"reward": reward,
                     "cash": self.env.cash,
                     "stock": self.env.stock,
                     "done": done})

            if done or (self.limit_step and step >= self.limit_step):
                break

        if self.logger:
            self.logger.close()

        return self.rewards
