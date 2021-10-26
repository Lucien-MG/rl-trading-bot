#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import gym
import gym_stock_exchange
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

from flask import *
from agent.random import AgentRandom

app = Flask(__name__)
app.config['SECRET_KEY'] = "random string"

env = gym.make("gym_stock_exchange:gym_stock_exchange-v0", stock_exchange_data_dir="data/cac40/")

agent = AgentRandom(env.action_space.n)

state = env.reset()
done = False

@app.route('/', methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        df = pd.read_csv('logs.csv', sep=",")
        df.iloc[:, 0].astype(float).plot()
        base_url = '/static/images/rewards.png'
        plt.savefig('src' + base_url)

        df.iloc[:, 2].astype(float).plot()
        base_url_cash = '/static/images/cash.png'
        plt.savefig('src' + base_url_cash)
        return render_template('reward.html', name='Rewards over time', url=base_url, url_cash=base_url_cash)
    return render_template('index.html')

while not done:
    action = agent.action(state)
    print(action)
    next_state, reward, done, info = env.step(action)

    state = next_state
    f = open("logs.csv", "a+")
    f.write(str(reward) + ',' + str(done) + ',' + str(info["cash"]) + '\n')
    f.close()
    print(state, reward, done, info)

if len(sys.argv) > 1 and sys.argv[1] == "--logs":
    app.run(host='localhost', port=5000)
