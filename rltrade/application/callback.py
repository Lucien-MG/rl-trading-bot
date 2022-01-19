#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑
import os

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import dash

from dash.dependencies import Input, Output, State
from dash import html

from application.css import *
from application.index_html import *

from agents.random import Agent

from reinforcement_api import train, list_env, run

from tools.utils import *


def add_callbacks(app):

    @app.callback(
        Output('reward', 'figure'),
        Output('money', 'figure'),
        Input('submit_button_2', 'n_clicks'),
        State('Agent', 'value'),
        State('Stock', 'value'),
        State('folder_value', 'value'),
        State('dqn', 'children'),
        State('random', 'children'))
    def call_agent(n_clicks, agent_value, stock_value, folder_value, dqn_values, rdm_values):
        if not n_clicks:
            return {'data': []}, {'data': []}
        
        folder = folder_value

        if not os.path.isdir(folder):
            os.mkdir(folder)
       
        if agent_value == 'dqn_v1': 
            save_dqn_config(dqn_values, folder)
        elif agent_value =='random':
            save_dqn_config(rdm_values, folder)
            #save_random_config(rdm_values, folder)

        print(stock_value)
        # THREAD 
        train(stock_value, agent_value, agent_config=folder + '/config.yaml', nb_episode=2, log_path=folder + '/logs.csv')

        # TODO pb on affiche que le dernier épisode
        

        df = pd.read_csv(folder + "/logs.csv", header=None, names= ['reward', 'done', 'money', 'action'], delimiter=",")
        fig = px.line(df, y="reward", hover_name="reward")
        fig.update_layout(title_text='Reward result', title_x=0.5)

        fig_cash = px.line(df, y="money", hover_name="money")
        fig_cash.update_layout(title_text='Money result', title_x=0.5)

        return fig, fig_cash
    

    @app.callback(
        Output('reward_prod', 'figure'),
        Output('money_prod', 'figure'),
        Output('text_folder_prod', 'style'),
        Input('submit_button_2_prod', 'n_clicks'),
        State('Agent_prod', 'value'),
        State('Stock_prod', 'value'),
        State('folder_value_prod', 'value'))
    def call_agent_prod(n_clicks, agent_value, stock_value, folder_value):
        if not n_clicks:
            return {'data': []}, {'data': []}, HIDDEN
        
        folder = folder_value
        
        if not os.path.isdir(folder):
            return {'data': []}, {'data': []}, WRONG_TEXT
       

        run(stock_value, agent_value, folder + '/config.yaml', folder + '/logs.csv')

        # logs_prod
        df = pd.read_csv(folder + "/logs.csv", header=None, names= ['reward', 'done', 'money', 'action'], delimiter=",")
        fig = px.line(df, y="reward", hover_name="reward")
        fig.update_layout(title_text='Reward result', title_x=0.5)

        fig_cash = px.line(df, y="money", hover_name="money")
        fig_cash.update_layout(title_text='Money result', title_x=0.5)

        return fig, fig_cash, HIDDEN




    @app.callback(
        Output('submit_button', 'style'),
        Output('Agent', 'disabled'),
        Output('Stock', 'disabled'),
        Output('folder', 'style'),
        Output('dqn', 'style'),
        Output('random', 'style'),
        Output('submit_button_2', 'style'),
        Input('submit_button', 'n_clicks'),
        State('Agent', 'value'))
    def show_settings(n_clicks, agent_value):
        if not n_clicks:
            return  NONE, False, False, HIDDEN, HIDDEN, HIDDEN, HIDDEN
        print(agent_value) 
        if agent_value == 'dqn_v1':
            return HIDDEN, True, True, NONE, NONE, HIDDEN, NONE
        elif agent_value == 'random':
            return HIDDEN, True, True, NONE, HIDDEN, NONE, NONE
        return HIDDEN, True, True, NONE, HIDDEN, HIDDEN, NONE
    

    @app.callback(
        Output('submit_button_prod', 'style'),
        Output('Agent_prod', 'disabled'),
        Output('Stock_prod', 'disabled'),
        Output('folder_prod', 'style'),
        Output('submit_button_2_prod', 'style'),
        Input('submit_button_prod', 'n_clicks'))
    def show_settings_prod(n_clicks):
        if not n_clicks:
            return  NONE, False, False, HIDDEN, HIDDEN
        return HIDDEN, True, True, NONE, NONE
    

    @app.callback(
        Output('sidebar', 'style'),
        Output('content', 'style'),
        Input('hide', 'n_clicks'),
        Input('show', 'n_clicks'))
    def update_style(hide_n, show_n):
        ctx = dash.callback_context

        if not ctx.triggered or ctx.triggered[0]['value'] == 0:
            return SIDEBAR_STYLE, CONTENT_STYLE
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'hide':
                return HIDDEN, CONTENT_STYLE_HIDDEN
            return SIDEBAR_STYLE, CONTENT_STYLE
    

    @app.callback(
        Output('sidebar_prod', 'style'),
        Output('content_prod', 'style'),
        Input('hide_prod', 'n_clicks'),
        Input('show_prod', 'n_clicks'))
    def update_style_prod(hide_n, show_n):
        ctx = dash.callback_context

        if not ctx.triggered or ctx.triggered[0]['value'] == 0:
            return SIDEBAR_STYLE, CONTENT_STYLE
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'hide_prod':
                return HIDDEN, CONTENT_STYLE_HIDDEN
            return SIDEBAR_STYLE, CONTENT_STYLE

    @app.callback(
        Output('page-content', 'children'),
        Input('url', 'pathname'))
    def display_page(pathname):
        if pathname == '/':
            return page_train
        elif pathname == '/product':
            return page_product
        elif pathname == '/data':
            return page_data
        else:
            return page_404
    
    @app.callback(
        Output('Info_Agent', 'children'),
        Input('Agent', 'value'))
    def display_info(value):
        if value == 'dqn_v1':
            return info_dqn
        return info_random
        
    @app.callback(
        Output('Data_fig', 'figure'),
        Input('Data', 'value'))
    def display_bourse(value):
        path = "data/" + value + ".csv"
        df = pd.read_csv(path, delimiter=",")
        print(df)
        fig = go.Figure(data=[go.Candlestick(x=df['date'],
                open=df['opening'],
                high=df['max'],
                low=df['min'],
                close=df['closing'])])
        return fig