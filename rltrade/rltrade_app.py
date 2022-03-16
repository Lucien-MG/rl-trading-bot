import uuid

from flask import Flask
from flask_login import LoginManager
from flask_bootstrap import Bootstrap

from app.blueprints.auth import auth as auth_blueprint
from app.blueprints.user import user as user_blueprint

from app.database.database import database
from app.database.models.user import User

from definitions import *


def create_database(app):
    database.init_app(app)
    database.create_all(app=app)


def create_login_manager(app):
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table,
        # use it in the query for the user
        return User.query.get(int(user_id))


def create_app():
    app = Flask(__name__, template_folder=TEMPLATE_DIRECTORY,
                static_folder=STATIC_DIRECTORY)

    Bootstrap(app)

    app.config['SECRET_KEY'] = uuid.uuid4().hex
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

    create_database(app)
    create_login_manager(app)

    # blueprint for auth routes in our app
    app.register_blueprint(auth_blueprint)

    # blueprint for non-auth parts of app
    app.register_blueprint(user_blueprint)

    return app


def launch(config):
    app = create_app()
    app.run(host="127.0.0.1", port=5001, debug=True)


if __name__ == "__main__":
    launch(None)
