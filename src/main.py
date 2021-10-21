#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import gym
import gym_stock_exchange

env = gym.make("gym_stock_exchange:gym_stock_exchange-v0", stock_exchange_data_dir="data/cac40/")

env.reset()
done = False

while not done:
    state, reward, done, info = env.step(1)
    print(state, reward, done, info)

