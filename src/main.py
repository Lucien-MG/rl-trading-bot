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
    global state, done
    if request.method == "POST":
        if request.form['submit_button'] == "view_value":
            df = pd.read_csv('logs.csv', sep=",")
            df.iloc[:, 0].astype(float).plot()
            base_url = '/static/images/rewards.png'
            plt.savefig('src' + base_url)
            plt.clf()
            df.iloc[:, 2].astype(float).plot()
            base_url_cash = '/static/images/cash.png'
            plt.savefig('src' + base_url_cash)
            plt.clf()
            return render_template('reward.html',
                                   name='Rewards over time',
                                   url=base_url,
                                   url_cash=base_url_cash)
        else:
           if os.path.exists("logs.csv"):
             os.remove("logs.csv")

           run(done, state)
           done = False
           state = env.reset()
           # agent = request.form['agent']
           return render_template('index.html', finish=True)
    return render_template('index.html', finish=False)

def run(done, state):
    while not done:
        action = agent.action(state)
        print(action)
        next_state, reward, done, info = env.step(action)

        state = next_state
        f = open("logs.csv", "a+")
        f.write(str(reward) + ',' + str(done) + ',' + str(info["cash"]) + ',' + str(action) +'\n')
        f.close()
        print(state, reward, done, info)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        app.run(host='localhost', port=5000)
    else:
        run(done, state)
