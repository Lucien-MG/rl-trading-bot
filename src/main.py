#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import gym
import gym_stock_exchange

from agent.random import AgentRandom

env = gym.make("gym_stock_exchange:gym_stock_exchange-v0", stock_exchange_data_dir="data/cac40/")

agent = AgentRandom(env.action_space.n)

state = env.reset()
done = False

while not done:
    action = agent.action(state)
    print(action)
    next_state, reward, done, info = env.step(action)

    state = next_state
    print(state, reward, done, info)

