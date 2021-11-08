#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

from .css import *

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

        dcc.Dropdown(id='Agent', 
                     options=[  {'label': 'Random', 'value': 'RDM'},
                                {'label': 'Deiss', 'value': 'DSS'},
                                {'label': 'Maitre Lucien', 'value': 'ML'},
                                {'label': 'Oui maitre', 'value': 'OM'}], value='RDM'),
        html.Br(),

        dcc.Dropdown(id='Stock', 
                     options=[  {'label': 'Paris', 'value': 'P'},
                                {'label': 'Wall Street', 'value': 'WL'}], value='P'),
        html.Br(),

        html.Div([
            dbc.Button(
                id='submit_button',
                n_clicks=0,
                children='Submit',
                color='primary'),
            ], className="d-grid gap-2 col-6 mx-auto")
	   
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


layout = html.Div([sidebar, content])
