
import signal
import sys
import os
import flask
import sqlite3
from flask_login import LoginManager, login_required, login_user, current_user, logout_user
from poc.flask.models import users, blogs


def signal_handler_callback(sig, frame):
    print('')
    print('Signal %s Frame %s' % (sig, frame))
    print('Exiting program !!!')
    sys.exit(0)


application = flask.Flask(__name__)
login = LoginManager(application)
conn = sqlite3.connect(':memory:', check_same_thread=False)


@application.route('/', methods=['GET'])
def welcome():
    html_text = """
    <html>
    <head>
        <meta charset="utf-8">
        <link rel="stylesheet" type="text/css" href="css/style.css">
    </head>
    <body>
    <h3>Welcome to Flask !!! </h3>
    <div id='welcome-page'>
        <a href="/login">Login Page</a>
        <a href="/profile">Profile Page</a>
        <a href="/home">Home Page</a>
        <a href="/signup">Signup Page</a>
        <a href="/profile">Profile Page</a>
        <a href="/settings">Settings Page</a>
    </div>
    """

    for h in flask.request.headers:
        html_text = html_text + '<br/>' + str(h[0]) + ': ' + str(h[1])
    html_text = html_text + """
    </body>
    </html>
    """
    return html_text


@login.user_loader
def load_user(userid):
    result = statement.execute('SELECT * FROM users where userid="{}"'.format(userid))
    row = result.fetchone()
    if row is None:
        return
    return users.User(*row[0: 4])


@application.route('/favicon.ico')
def favicon():
    return flask.send_from_directory(os.path.join(application.root_path, 'static'),
                                     'favicon.png', mimetype='image/vnd.microsoft.icon')


@application.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    print("Path: {} Method: {}".format(request.path, request.method))
    return flask.jsonify({'tasks': tasks})


@application.errorhandler(404)
def not_found(error):
    print(error)
    return flask.make_response(flask.jsonify({'error': 'Sorry!!! Url you entered is not found'}), 404)


@application.errorhandler(500)
def internal_error(error):
    print(error)
    return flask.make_response(flask.jsonify({'error': 'Internal Error Occurred'}), 500)


@application.route('/login', methods=['GET'])
def get_login():
    return flask.render_template('login.html')


@application.route('/login', methods=['POST'])
def post_login():
    if current_user.is_authenticated:
        return flask.redirect(flask.url_for('home'))

    user = users.User.get_valid_user(conn, flask.request.form['userid'], flask.request.form['userid'],
                                     flask.request.form['pass'])
    if user is not None:
        login_user(user)
        response = flask.make_response(flask.redirect('/home'))
        return response
    else:
        return flask.render_template('login.html', error='error')


@application.route('/home')
@login_required
def home():
    return flask.render_template('home.html')


@application.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return flask.redirect(flask.url_for('get_login'))


@application.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if flask.request.method == 'GET':
        return flask.render_template('profile.html')
    else:
        if flask.request.form['action'] == 'Cancel':
            return flask.redirect(flask.url_for('home'))
        else:
            updated_user = users.User(flask.request.form['fname'], flask.request.form['lname'], None, None)
            updated_user.update(conn, current_user.userid)
            return flask.redirect(flask.url_for('home'))


@application.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    return flask.render_template('settings.html')


@application.route('/signup', methods=['GET'])
def get_signup():
    return flask.render_template('signup.html')


@application.route('/signup', methods=['POST'])
def post_signup():
    user = users.User(flask.request.form['fname'], flask.request.form['lname'], flask.request.form['email'],
                      flask.request.form['userid'])
    user.create(conn, flask.request.form['pass'])
    return flask.redirect('/login')


def get_file(filename):
    try:
        src = os.path.join(os.path.abspath(os.path.dirname(__file__)) + '/static', filename)
        return open(src).read()
    except IOError as exc:
        return str(exc)


@application.route('/<path:path>')
def get_resource(path):
    mime_types = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript",
    }
    file_extension = os.path.splitext(path)[1]
    if file_extension == '.map':
        return flask.Response()
    content = get_file(path)
    return flask.Response(content, mimetype=mime_types[file_extension])


@application.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.set_cookie('', '', secure=True, httponly=True)
    return response


@application.route('/saveOrUpdateBlog', methods=['POST'])
def save_or_update_blog():
    blog = blogs.Blogs(flask.request.form['id'], flask.request.form['user'], flask.request.form['blog_sub'],
                       flask.request.form['blog'], flask.request.form['time'])
    blog.create(conn)
    return flask.make_response(flask.jsonify({'result': 'Success'}), 200)


def main():
    global statement
    statement = conn.cursor()
    statement.execute('CREATE TABLE blogs("identifier", "user", "blog_sub", "blog", "time")')
    statement.execute('CREATE TABLE users("fname", "lname", "email", "userid", "pass")')
    statement.execute('INSERT INTO users VALUES(?, ?, ?, ?, ?)',
                      ['Test', 'Test Surname', 'test@gmail.com', 'tid', 'tpass'])
    login.login_view = 'get_login'
    application.secret_key = 'super secret key'
    application.run()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler_callback)
    main()
