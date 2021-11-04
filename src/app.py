#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from application.css import *

from application.callback import call_agent
from application.html import sidebar, content

def run_app():
    app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = dash.html.Div([sidebar, content])
    app.run_server(debug=True)
