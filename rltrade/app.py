import uuid

from flask import Flask, url_for, redirect, render_template, session, request
from flask_login import LoginManager
from flask_bootstrap import Bootstrap

from application.database import database

from definitions import *

from core import core
from config.environment.environment_config import EnvironmentConfig

cache = {}

def create_app():
    app = Flask(__name__)
    bootstrap = Bootstrap(app)

    app.config['SECRET_KEY'] = uuid.uuid4().hex
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    database.init_app(app)

    from application.models import User

    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        return User.query.get(int(user_id))

    # blueprint for auth routes in our app
    from application.blueprints.auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    # blueprint for non-auth parts of app
    from application.blueprints.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app, bootstrap, database

app, bootstrap, database = create_app()

database.create_all(app=app)

@app.route("/welcome")
def welcome():
    return render_template("welcome.html")


@app.route("/user/home")
def home():
    return render_template("/user/home.html")


@app.route("/user/stocks", methods=['GET', 'POST'])
def stocks():
    if request.method == 'POST':
        default_value = None
        stock_index = request.form.get('stockIndex', default_value)

        env_config = EnvironmentConfig(
                        environment_config_path=None,
                        index = stock_index,
                        start_date =  "2022-01-12",
                        end_date = "2022-02-06")
        
        env = core.create_environment(env_config)

        labels = list(env.env.data['date'].dt.strftime('%Y-%m-%d:%r'))
        values = list(env.env.data['close'])

        cache['/user/stocks'] = {'labels': labels, 'values': values}
    elif request.method == 'GET':
        labels = cache['/user/stocks']['labels'] if '/user/stocks' in cache else None
        values = cache['/user/stocks']['values'] if '/user/stocks' in cache else None
    return render_template("user/stocks.html", values=values, labels=labels)


@app.route("/user/training", methods=['GET', 'POST'])
def training():
    labels = ["January", "February", "March",
              "April", "May", "June", "July", "August"]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    return render_template("user/training.html", values=values, labels=labels)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
