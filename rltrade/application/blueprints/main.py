from collections import defaultdict

from flask import Blueprint, render_template, redirect, url_for, request
from flask_login import login_required, current_user

from core import core
from config.environment.environment_config import EnvironmentConfig

DEFAULT_FORM_VALUE = None

cache = defaultdict(lambda: defaultdict(lambda: None))
main = Blueprint('main', __name__)


@main.route('/')
@login_required
def index():
    return redirect('/user/home')


@main.route('/user/home')
@login_required
def home():
    return render_template("/user/home.html", user=current_user)


@main.route('/user/stocks')
@login_required
def stocks():
    return render_template("/user/stocks.html", user=current_user, meta=cache[current_user.name]['stocks'])


@main.route('/user/stocks', methods=['POST'])
@login_required
def stocks_post():
    stock_index = request.form.get('stockIndex', DEFAULT_FORM_VALUE)
    start_date = request.form.get('startDate', DEFAULT_FORM_VALUE)
    end_date = request.form.get('endDate', DEFAULT_FORM_VALUE)

    if stock_index is None or stock_index == "":
        return redirect('/user/stocks')

    env_config = EnvironmentConfig(
        environment_config_path=None,
        index=stock_index,
        start_date=start_date,
        end_date=end_date)

    env = core.create_environment(env_config)

    labels = list(env.env.data['date'].dt.strftime('%Y-%m-%d:%r'))
    values = list(env.env.data['close'])

    cache[current_user.name]['stocks'] = {'index': stock_index, 'labels': labels, 'values': values}

    return render_template("/user/stocks.html", user=current_user, meta=cache[current_user.name]['stocks'])


@main.route('/user/training')
@login_required
def training():
    labels = ["January", "February", "March",
              "April", "May", "June", "July", "August"]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    return render_template("/user/training.html", user=current_user, values=values, labels=labels)

@main.route('/user/routines')
@login_required
def routines():
    labels = ["January", "February", "March",
              "April", "May", "June", "July", "August"]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    return render_template("/user/routines.html", user=current_user)
