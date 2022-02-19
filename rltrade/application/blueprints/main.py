from flask import Blueprint, render_template, redirect, url_for, request
from flask_login import login_required, current_user

from core import core
from config.environment.environment_config import EnvironmentConfig

cache = {}
main = Blueprint('main', __name__)


@main.route('/')
@login_required
def index():
    return redirect('/user/home')


@main.route('/user/home')
@login_required
def home():
    return render_template("/user/home.html", name=current_user.name)


@main.route('/user/stocks')
@login_required
def stocks():
    labels = cache['/user/stocks']['labels'] if '/user/stocks' in cache else None
    values = cache['/user/stocks']['values'] if '/user/stocks' in cache else None

    return render_template("/user/stocks.html", name=current_user.name, values=values, labels=labels)


@main.route('/user/stocks', methods=['POST'])
@login_required
def stocks_post():
    default_value = None
    stock_index = request.form.get('stockIndex', default_value)
    start_date = request.form.get('startDate', default_value)
    end_date = request.form.get('endDate', default_value)

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

    cache['/user/stocks'] = {'labels': labels, 'values': values}

    return render_template("/user/stocks.html", name=current_user.name, values=values, labels=labels)


@main.route('/user/training')
@login_required
def training():
    labels = ["January", "February", "March",
              "April", "May", "June", "July", "August"]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    return render_template("/user/training.html", name=current_user.name, values=values, labels=labels)

@main.route('/user/routines')
@login_required
def routines():
    labels = ["January", "February", "March",
              "April", "May", "June", "July", "August"]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    return render_template("/user/routines.html", name=current_user.name)
