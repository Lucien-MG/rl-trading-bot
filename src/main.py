#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import gym
import gym_stock_exchange
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

from agent.random import AgentRandom

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.express as px

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}

env = gym.make("gym_stock_exchange:gym_stock_exchange-v0", stock_exchange_data_dir="data/cac40/")

agent = AgentRandom(env.action_space.n)

state = env.reset()
done = False

sidebar = html.Div(
    [
        html.H2('Our agents', style=TEXT_STYLE),
        html.Hr(),

        dcc.Dropdown(id='my-dropdown', 
                     options=[  {'label': 'Random', 'value': 'RDM'},
                                {'label': 'Deiss', 'value': 'DSS'},
                                {'label': 'Maitre Lucien', 'value': 'ML'},
                                {'label': 'Oui maitre', 'value': 'OM'}], value='RDM'),
        html.Br(),

        html.Div(
            [
                dbc.Button(
                    id='submit_button',
                    n_clicks=0,
                    children='Submit',
                    color='primary'),

            ],
            className="d-grid gap-2 col-6 mx-auto",),

    ],
    style=SIDEBAR_STYLE,
)


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([sidebar])


@app.callback(
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value')])
def call_agent(n_clicks, dropdown_value):
    global done, state
    if os.path.exists("logs.csv"):
        os.remove("logs.csv")
    run(done, state)
    done = False
    state = env.reset()

    df = pd.read_csv("logs.csv", header=None, names= ['reward', 'done', 'money', 'action'], delimiter=",")
    fig = px.line(df, y="reward", hover_name="reward")
    fig.update_layout(title_text='Reward result', title_x=0.5)

    fig_cash = px.line(df, y="money", hover_name="money")
    fig_cash.update_layout(title_text='Money result', title_x=0.5)

    return fig
    # df = pd.read_csv('logs.csv', sep=",")
    # df.iloc[:, 0].astype(float).plot()
    # base_url = '/static/images/rewards.png'
    # plt.savefig('src' + base_url)
    # plt.clf()
    # df.iloc[:, 2].astype(float).plot()
    # base_url_cash = '/static/images/cash.png'
    # plt.savefig('src' + base_url_cash)
    # plt.clf()

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
        app.run_server()
    else:
        run(done, state)
