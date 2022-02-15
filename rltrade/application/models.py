from flask_login import UserMixin
from application.database import database

class User(database.Model, UserMixin):
        id = database.Column(database.Integer, primary_key=True) # primary keys are required by SQLAlchemy
        email = database.Column(database.String(100), unique=True)
        password = database.Column(database.String(100))
        name = database.Column(database.String(1000))
