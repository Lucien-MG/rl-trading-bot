import uuid

from flask import Flask, url_for, redirect, render_template, session, request
from flask_login import LoginManager
from flask_bootstrap import Bootstrap

from application.database import database

from definitions import *

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

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
