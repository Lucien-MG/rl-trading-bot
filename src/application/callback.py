#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑
import os
import pandas as pd
import plotly.express as px

import dash
from dash.dependencies import Input, Output, State

from application.css import SIDEBAR_STYLE, SIDEBAR_STYLE_HIDDEN
from application.css import CONTENT_STYLE, CONTENT_STYLE_HIDDEN

from agents.random import Agent
from reinforcement_api import train

#env = gym.make("gym_stock_exchange:gym_stock_exchange-v0", stock_exchange_data_dir="data/cac40/")

agent = None

#state = env.reset()
done = False

def add_callbacks(app):

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


    @app.callback(
        Output('reward', 'figure'),
        Output('money', 'figure'),
        Input('submit_button', 'n_clicks'),
        State('Agent', 'value'),
        State('Stock', 'value'))
    def call_agent(n_clicks, agent_value, stock_value):
        if not n_clicks:
            return {'data': []}, {'data': []}
        global done, state
        if os.path.exists("logs.csv"):
            os.remove("logs.csv")
        done = False
        #state = env.reset()
        
        train(stock_value, agent_value, 'agent_config.yaml', 2)

        # TODO pb on affiche que le dernier épisode
        
        #run(dropdown_agent)

        #if dropdown_value == 'RDM':
        #    agent = Agent(env.action_space.n)
        #elif dropdown_value == 'DQN':
            #agent = Agent( blabla )
        #agent = AgentRandom(env.action_space.n)
        
        #run(done, state)

        df = pd.read_csv("logs.csv", header=None, names= ['reward', 'done', 'money', 'action'], delimiter=",")
        fig = px.line(df, y="reward", hover_name="reward")
        fig.update_layout(title_text='Reward result', title_x=0.5)

        fig_cash = px.line(df, y="money", hover_name="money")
        fig_cash.update_layout(title_text='Money result', title_x=0.5)

        return fig, fig_cash


    @app.callback(
        [
            Output('sidebar', 'style'),
            Output('content', 'style')
        ],
        [
            Input('hide', 'n_clicks'),
            Input('show', 'n_clicks')
        ])
    def update_style(hide_n, show_n):
        ctx = dash.callback_context

        if not ctx.triggered:
            return [SIDEBAR_STYLE, CONTENT_STYLE]
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'hide':
                return [SIDEBAR_STYLE_HIDDEN, CONTENT_STYLE_HIDDEN]
            return [SIDEBAR_STYLE, CONTENT_STYLE]

