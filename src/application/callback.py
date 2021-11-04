#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import dash
from dash.dependencies import Input, Output, State

def run(done, state):
    while not done:
        action = agent.action(state)
        print(action)
        next_state, reward, done, info = env.step(action)

        print(info)
        state = next_state
        f = open("logs.csv", "a+")
        f.write(str(reward) + ',' + str(done) + ',' + str(info["cash"]) + ',' + str(action) +'\n')
        f.close()
        print(state, reward, done, info)

def call_agent(n_clicks, dropdown_value):
    if not n_clicks:
        return {'data': []}, {'data': []}
    global done, state
    if os.path.exists("logs.csv"):
        os.remove("logs.csv")
    done = False
    state = env.reset()
    run(done, state)

    df = pd.read_csv("logs.csv", header=None, names= ['reward', 'done', 'money', 'action'], delimiter=",")
    fig = px.line(df, y="reward", hover_name="reward")
    fig.update_layout(title_text='Reward result', title_x=0.5)

    fig_cash = px.line(df, y="money", hover_name="money")
    fig_cash.update_layout(title_text='Money result', title_x=0.5)

    return fig, fig_cash
