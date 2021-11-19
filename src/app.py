#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import dash
import dash_bootstrap_components as dbc

from application.callback import add_callbacks
from application.css import *

from application.index_html import add_layout
import reinforcement_api

def run_app():
    app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = add_layout(reinforcement_api)
    add_callbacks(app)
    app.run_server(debug=True)
