#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

from .css import *

sidebar = html.Div(
    [
        html.H2('Our agents', style=TEXT_STYLE),
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

content = html.Div(
  [
    dcc.Loading(id='loading',type='circle',
	children=[
		html.Div([
    			dcc.Graph(id='reward'), 
    			html.Br(),
    			dcc.Graph(id='money')
  			]
			,style=CONTENT_STYLE)
		],
        style=LOADING_STYLE
	)
  ]
)
