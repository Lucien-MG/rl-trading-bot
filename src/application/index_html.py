#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

from .css import *

from tools.utils import load_config

def add_layout(reinforcement_api):
    
    agents = [{'label': agent.capitalize(), 'value': agent} for agent in reinforcement_api.list_agent()]
    envs = [{'label': env.capitalize(), 'value': env} for env in reinforcement_api.list_env()]
    config = load_config('agent_config.yaml')
    dqn = []
    for i in list(config):
        if i =='parameters':
            parameters = config[i]
            for j in list(parameters):
                dqn.append(
                    dbc.Row([
                        html.Div(j, style=AUTO),
                        dcc.Input(value = parameters[j], style=AUTO, maxLength=10, size='2')
                    ], style={'display':'flex','align-item':'center','justify-content':'space-between'}))
                #dqn.append(html.Br())
        else:
            dqn.append(
                dbc.Row([
                    html.Div(i, style=AUTO),
                    dcc.Input(value = config[i], style=AUTO, maxLength=10, size='2')
                ], style={'display':'flex','align-item':'center','justify-content':'space-between'}))
            #dqn.append(html.Br())

    sidebar = html.Div([
	    	html.Div([
		    	html.Div([
			    	html.H2('Our agents', style=AGENT_STYLE)
				    ], style = {
					    'width': '95%',
					    'margin-left': '5%'}),
			    
                html.Div([
				    dbc.Button(
                        id='hide',
                        n_clicks=0,
                        children="Hide",
                        color='primary')
                    ], style = {
                        'width': '5%',
                        'position': 'relative',
                        'left': '-3em'}),
                ], style = {
                    'content': '',
                    'display': 'flex',
                    'clear': 'both',
                    'align-items': 'center',
                    'justify-content': 'space-between'}),
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

            html.Div(
                dbc.Row([
                    html.Div('Folder', style=AUTO),
                    dcc.Input(value = '', style=AUTO, maxLength=10, size='7')
                ], style = ROW_SIDEBAR
            ), style = HIDDEN, id = 'folder'),
            

            html.Br(),

            html.Div(dqn, id='dqn', style = HIDDEN),

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
		
                    dcc.Tab(label='Agent', children=[
			            html.H2('Some informations about our agent !')
		                ], style=tab_style, selected_style=tab_selected_style)
	                ])
                ], style=CONTENT_STYLE, id='content')
            ])


    return html.Div([sidebar, content])
