#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

class RunEnvironment:
    """ Initialize the RunEnv class.
    Args:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
    Attributes:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
    """
    def __init__(self, env, agent, render=None):
        self.env = env
        self.agent = agent
        self.render = render

    def step(self, state):
        # agent choose action with state observation
        action = self.agent.action(state)

        # agent performs the selected action
        next_state, reward, done, info = self.env.step(action)

        # agent performs internal updates based on sampled experience
        self.agent.step(state, action, reward, next_state, done)

        # Return results
        return next_state, reward, done, info

    def episode(self):
        # begin the episode
        step = 0
        rewards = []
        done = False
        state = self.env.reset()

        while not done:
            # Use the render mode if needed
            if self.render:
                self.env.render()

            # Execute one step: the agent take an action and the env return the new state
            next_state, reward, done, info = self.step(state)

            # update the sampled reward
            rewards.append(reward)

            # update the state (s <- s') to next time step
            state = next_state

            # Next step
            step += 1

        return rewards
