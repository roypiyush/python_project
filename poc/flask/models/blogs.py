
class Blogs:
    
    def __init__(self, identifier, user, blog_sub, blog, time):
        self.identifier = identifier
        self.user = user
        self.blog_sub = blog_sub
        self.blog = blog
        self.time = time

    def __repr__(self) -> str:
        return "Blog >>> {} {} {} {} {}".format(self.identifier, self.user, self.blog_sub, self.blog, self.time)

    def create(self, connection):
        statement = connection.cursor()
        statement.execute('INSERT INTO blogs VALUES(?, ?, ?, ?, ?)',
                          [self.identifier, self.user, self.blog_sub, self.blog, self.time])
        connection.commit()

    def update(self, connection):
        statement = connection.cursor()
        statement.execute('UPDATE blogs SET identifier=?, user=?, blog_sub=?, blog=?, time=?',
                          [self.identifier, self.user, self.blog_sub, self.blog, self.time])
        connection.commit()
