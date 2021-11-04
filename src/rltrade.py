#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import importlib

import gym
import gym_stock_exchange
import sys

import pandas as pd
import matplotlib.pyplot as plt
import os

import dash

import plotly.express as px

from app import run_app

env = gym.make("gym_stock_exchange:gym_simulation_stock_exchange-v0", stock_exchange_data_dir="data/cac40/")

agent = AgentRandom(env.action_space.n)

state = env.reset()
done = False

if __name__ == '__main__':
    run_app()
