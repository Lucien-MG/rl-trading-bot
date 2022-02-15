from flask import Blueprint, render_template, redirect, url_for, request
from flask_login import login_required, current_user

main = Blueprint('main', __name__)

@main.route('/')
@login_required
def index():
    return 'Index'

@main.route('/home')
@login_required
def home():
    return render_template("/user/home.html", name=current_user.name)
