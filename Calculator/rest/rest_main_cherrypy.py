import signal
import cherrypy
from Calculator.impl import add_operation
from Calculator.impl import subtract_operation
from Calculator.impl import multiply_operation
from Calculator.impl import divide_operation

USERS = {'user': 'pass'}


def error_page(status, message, traceback, version):
    print(traceback)
    return str(dict(status=status, message=message, version=version))


class CalculatorController(object):

    @cherrypy.expose()
    def index(self):
        return "Calculator Service is Up"

    @cherrypy.expose()
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def add(self):
        try:
            request_data = getattr(cherrypy.request, "json", None)
            if request_data is None:
                raise KeyError("Missing JSON Body")
            a_num = request_data["a"]
            b_num = request_data["b"]
            op = add_operation.AddOperation(a_num, b_num)
            return dict(result=op.process())
        except (TypeError, KeyError) as e:
            cherrypy.response.status = 400
            return dict(errorMessage=str(e))

    @cherrypy.expose()
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def sub(self):
        try:
            request_data = getattr(cherrypy.request, "json", None)
            if request_data is None:
                raise KeyError("Request Invalid")
            a_num = request_data["a"]
            b_num = request_data["b"]
            op = subtract_operation.SubtractOperation(a_num, b_num)
            return dict(result=op.process())
        except (TypeError, KeyError) as e:
            cherrypy.response.status = 400
            return dict(errorMessage=str(e))

    @cherrypy.expose()
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def mul(self):
        try:
            request_data = cherrypy.request.json
            a_num = request_data["a"]
            b_num = request_data["b"]
            op = multiply_operation.MultiplyOperation(a_num, b_num)
            return dict(result=op.process())
        except (TypeError, KeyError) as e:
            cherrypy.response.status = 400
            return dict(errorMessage=str(e))

    @cherrypy.expose()
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def div(self):
        try:
            request_data = cherrypy.request.json
            a_num = request_data["a"]
            b_num = request_data["b"]
            op = divide_operation.DivideOperation(a_num, b_num)
            return dict(result=op.process())
        except (TypeError, KeyError) as e:
            cherrypy.response.status = 400
            return dict(errorMessage=str(e))


def validate_password(realm, username, password):
    print("%s", realm)
    if username in USERS and USERS[username] == password:
        return True
    return False


def start_server():
    conf = {
        '/': {
            'tools.gzip.on': True,
            'tools.auth_basic.on': True,
            'tools.auth_basic.realm': 'localhost',
            'tools.auth_basic.checkpassword': validate_password
        }
    }
    cherrypy.tree.mount(CalculatorController(), '/', conf)
    cherrypy.config.update({'error_page.404': error_page})
    cherrypy.config.update({'error_page.400': error_page})
    cherrypy.config.update({'error_page.401': error_page})
    cherrypy.config.update({'error_page.500': error_page})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()


def signal_handler_callback(sig, frame):
    print('')
    print('Signal %s Frame %s' % (sig, frame))
    print('Exiting program !!!')
    cherrypy.engine.stop()
    cherrypy.engine.graceful()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler_callback)
    start_server()
