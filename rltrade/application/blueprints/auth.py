from application.database import database

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user
from werkzeug.security import generate_password_hash, check_password_hash

from application.models import User

auth = Blueprint('auth', __name__)

@auth.route('/login')
def login():
    return render_template("login.html")

@auth.route('/login', methods=['POST'])
def login_post():
    # login code goes here
    email = request.form.get('email')
    password = request.form.get('password')

    user = User.query.filter_by(email=email).first()

    # check if the user actually exists
    # take the user-supplied password, hash it, and compare it to the hashed password in the database
    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('auth.login')) # if the user doesn't exist or password is wrong, reload the page
    
    login_user(user, remember=True)

    # if the above check passes, then we know the user has the right credentials
    return redirect(url_for('main.home'))

@auth.route('/signup')
def signup():
    return render_template("signup.html")

@auth.route('/signup', methods=['POST'])
def signup_post():
    # code to validate and add user to database goes here
    email = request.form.get('email')
    name = request.form.get('name')
    password = request.form.get('password')

    user = User.query.filter_by(email=email).first() # if this returns a user, then the email already exists in database

    if user: # if a user is found, we want to redirect back to signup page so user can try again
        flash('User already exists.')
        return redirect(url_for('auth.signup'))

    # create a new user with the form data. Hash the password so the plaintext version isn't saved.
    new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))

    # add the new user to the database
    database.session.add(new_user)
    database.session.commit()

    return redirect(url_for('auth.login'))

@auth.route('/welcome')
def welcome():
    return render_template("welcome.html")

@auth.route('/logout')
def logout():
    return 'Logout'
