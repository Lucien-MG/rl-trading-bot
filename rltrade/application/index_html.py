#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

from .css import *

from core.utils.utils import load_config

from core import core

agents = [{'label': agent.capitalize(), 'value': agent} for agent in core.list_agent()]
envs = [{'label': env.capitalize(), 'value': env} for env in core.list_env()]
datas = [{'label': data.capitalize(), 'value': data} for data in core.list_data()]

rltrade_config = load_config('config/rltrade_config.yaml')
#random_config = load_config('config.yaml')

#dqn = []
#random = []

#for i in list(random_config):
'''    random.append(
        dbc.Row([
            html.Div(i, style=AUTO),
            dcc.Input(value = random_config[i], style=AUTO, maxLength=10, size='2')
        ], style = ROW_SIDEBAR)
    )

for i in list(dqn_config):
    if i =='parameters':
        parameters = dqn_config[i]
        for j in list(parameters):
            dqn.append(
                dbc.Row([
                    html.Div(j, style=AUTO),
                    dcc.Input(value = parameters[j], style=AUTO, maxLength=10, size='2')
                ], style = ROW_SIDEBAR))
    elif i == 'input_shape':
        dqn.append(
            dbc.Row([
                html.Div(i, style=AUTO),
                dcc.Input(value = dqn_config[i][0], style=AUTO, maxLength=10, size='2'),
                dcc.Input(value = dqn_config[i][1], style=AUTO, maxLength=10, size='2')
            ], style = ROW_SIDEBAR))
    else:
        dqn.append(
            dbc.Row([
                html.Div(i, style=AUTO),
                dcc.Input(value = dqn_config[i], style=AUTO, maxLength=10, size='2')
            ], style = ROW_SIDEBAR))
'''
sidebar = html.Div([
        html.Div([
		    html.Div([
			    html.H2('Our agents', style=AGENT_STYLE)
		    ], style = SIDEBAR_HEADER),
			    
                html.Div([
				    dbc.Button(
                        id='hide',
                        n_clicks=0,
                        children="Hide",
                        color='primary')
                    ], style = BUTTON_HIDE),
                ], style = ROW_SIDEBAR),
            html.Hr(),

            html.Div([dcc.Dropdown(id='Agent', options= agents, value=agents[0].get('value'), clearable=False, disabled=False),
            html.Br(),
            dcc.Dropdown(id='Stock', options= envs, value=envs[0].get('value'), clearable=False, disabled=False)]),
            html.Br(),
            html.Div([
                dbc.Button(
                    id='submit_button',
                    n_clicks=0,
                    children='Submit',
                    color='primary'),
                ], className="d-grid gap-2 col-6 mx-auto"),

            html.Br(),

            #'''html.Div(
            #    dbc.Row([
            #        html.Div('Folder', style=AUTO),
            #        dcc.Input(value = '', style=AUTO, maxLength=10, size='7', id='folder_value')
            #    ], style = ROW_SIDEBAR
            #), style = HIDDEN, id = 'folder'),'''
            


            #html.Div(dqn, id='random', style = HIDDEN), #random
            #html.Div(dqn, id='dqn', style = HIDDEN),

            html.Div([
                dbc.Button(
                    id='submit_button_2',
                    n_clicks=0,
                    children='Submit',
                    color='primary',
                    style=HIDDEN),
                ], className="d-grid gap-2 col-6 mx-auto"),

            html.Br(),
            dcc.Link("Go to product page", href='/product'),
            html.Br(),
            dcc.Link("Go to data page", href='/data')
            ],  style=SIDEBAR_STYLE, id='sidebar')

content = html.Div([
        html.Div([
            dbc.Button(
                id='show',
                n_clicks=0,
                children='Show',
                color='primary',
                style=SHOW_STYLE)]),

        html.Div([
            dcc.Tabs([
		        dcc.Tab(label='Result', children=[
			        dcc.Loading(id='loading', type='circle', children=[
				        html.Div([
    				        dcc.Graph(id='reward'), 
    				        html.Br(),
    				        dcc.Graph(id='money')])
			            ], style=LOADING_STYLE)
		            ], style=tab_style, selected_style=tab_selected_style),
		
                dcc.Tab(id = "Info_Agent", label='Agent', children=[
			        html.H2('Some informations about our agent !')
		            ], style=tab_style, selected_style=tab_selected_style)
	            ])], style=CONTENT_STYLE, id='content')])

sidebar_prod = html.Div([
        html.Div([
		    html.Div([
			    html.H2('Our agents', style=AGENT_STYLE)
		    ], style = SIDEBAR_HEADER),
			    
                html.Div([
				    dbc.Button(
                        id='hide_prod',
                        n_clicks=0,
                        children="Hide",
                        color='primary')
                    ], style = BUTTON_HIDE),
                ], style = ROW_SIDEBAR),
            html.Hr(),

            html.Div([dcc.Dropdown(id='Agent_prod', options= agents, value=agents[0].get('value'), clearable=False, disabled=False),
            html.Br(),
            dcc.Dropdown(id='Stock_prod', options= envs, value=envs[0].get('value'), clearable=False, disabled=False)]),
            html.Br(),
            html.Div([
                dbc.Button(
                    id='submit_button_prod',
                    n_clicks=0,
                    children='Submit',
                    color='primary'),
                ], className="d-grid gap-2 col-6 mx-auto"),

            html.Br(),

            html.Div(
                dbc.Row([
                    html.Div('Folder', style=AUTO),
                    dcc.Input(value = '', style=AUTO, maxLength=10, size='7', id='folder_value_prod')
                ], style = ROW_SIDEBAR
            ), style = HIDDEN, id = 'folder_prod'),

            html.P("This folder doesn't exist...",
                id="text_folder_prod",
                style=HIDDEN),

            html.Br(),

            html.Div([
                dbc.Button(
                    id='submit_button_2_prod',
                    n_clicks=0,
                    children='Submit',
                    color='primary',
                    style=HIDDEN),
                ], className="d-grid gap-2 col-6 mx-auto"),

            html.Br(),
            dcc.Link("Go to train page", href='/'),
            ],  style=SIDEBAR_STYLE, id='sidebar_prod')



content_prod =html.Div([
        html.Div([
            dbc.Button(
                id='show_prod',
                n_clicks=0,
                children='Show',
                color='primary',
                style=SHOW_STYLE)]),

        html.Div([
            dcc.Tabs([
		        dcc.Tab(label='Result', children=[
			        dcc.Loading(id='loading_prod', type='circle', children=[
				        html.Div([
    				        dcc.Graph(id='reward_prod'),
    				        html.Br(),
    				        dcc.Graph(id='money_prod')])
			            ], style=LOADING_STYLE)
		            ], style=tab_style, selected_style=tab_selected_style),
		
                dcc.Tab(id = "Info_Agent_prod", label='Agent', children=[
			        html.H2('Some informations about our agent !')
		            ], style=tab_style, selected_style=tab_selected_style)
	            ])], style=CONTENT_STYLE, id='content_prod')])


content_data = html.Div([
    dcc.Dropdown(id='Data', options= datas, value=datas[0].get('value'), clearable=False),
    html.Br(),
    dcc.Graph(id='Data_fig'),
    dcc.Link("Go to train page", href='/')])


layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')])

page_train = html.Div([sidebar, content])
page_product = html.Div([sidebar_prod, content_prod])
page_data = html.Div(content_data)
page_404 = html.Div([html.H1("404 not found"), dcc.Link("Go to train page", href='/')])

info_dqn = html.Div([
    html.H1("DQN", style=TITLE_TEXT),
    html.Br(),
    html.P("Q-learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state. It does not require a model of the environment (hence 'model-free'), and it can handle problems with stochastic transitions and rewards without requiring adaptions."),
    html.Br(),
    html.P("For any finite Markov decision progress (FMDP), Q-learning finds an optimal policy in the sense of maximizing the expected value of the total reward over ay and all successive steps, starting from the current state. Q-learning can identify and optimal action-selection policy for any given FMDP, given infinite exploration time and a partly-random policy. 'Q' refers to the funtion that the algorithm computes - the expected rewards for an action taken in a give state.")])

info_random = html.Div([
    html.H1("Random agent", style=TITLE_TEXT),
    html.Br(),
    html.P("This is a random agent taking random desicions based on the same probability (Buy/Sell/Skip) !")])
