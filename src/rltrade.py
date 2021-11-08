#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import importlib

import gym
import gym_stock_exchange
import sys

import pandas as pd
import matplotlib.pyplot as plt
import os

import dash

import plotly.express as px

from broker import run
from app import run_app
from tools.cmd import argument_parser

if __name__ == '__main__':
    args = argument_parser()
    if args.interactive:
        run_app()
    else:
        run()
