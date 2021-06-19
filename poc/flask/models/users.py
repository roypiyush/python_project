from flask_login import UserMixin


class User(UserMixin):
    def __init__(self, fname, lname, email, userid):
        self.fname = fname
        self.lname = lname
        self.userid = userid
        self.email = email

    def get_id(self):
        return self.userid

    def create(self, connection, password):
        statement = connection.cursor()
        statement.execute('INSERT INTO users VALUES(?, ?, ?, ?, ?)',
                          [self.fname, self.lname, self.email, self.userid, password])
        connection.commit()

    def update(self, connection, userid):
        statement = connection.cursor()
        statement.execute('UPDATE users SET fname=?, lname=? where userid=?',
                          [self.fname, self.lname, userid])
        connection.commit()

    @staticmethod
    def get_valid_user(connection, userid, email, password):
        statement = connection.cursor()
        result = statement.execute('SELECT * FROM users where (userid="{}" OR email="{}") AND pass="{}"'.format(
            userid, email, password))
        row = result.fetchone()
        if row is not None and userid in row:
            user = User(*row[0: 4])
            return user
        else:
            return None
