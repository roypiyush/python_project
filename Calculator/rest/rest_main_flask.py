from flask import Flask, request, jsonify, make_response
from Calculator.impl import add_operation
from Calculator.impl import subtract_operation
from Calculator.impl import multiply_operation
from Calculator.impl import divide_operation


app = Flask(__name__)


class Result(object):

    def __init__(self, result):
        self.result = result


@app.route('/')
def index():
    return "Calculator Service is Up !!!"


@app.route('/add', methods=["POST"])
def add_service():

    try:
        request_data = request.json
        a_num = request_data["a"]
        b_num = request_data["b"]
        add = add_operation.AddOperation(a_num, b_num)

        resp = make_response(jsonify(result=add.process()))
        return resp
    except (TypeError, KeyError) as e:
        resp = make_response(jsonify(result=str(e)))
        resp.content_type = 'application/json'
        resp.status_code = 400
        return resp


@app.route('/sub', methods=["POST"])
def sub_service():
    try:
        request_data = request.json
        a_num = request_data["a"]
        b_num = request_data["b"]
        op = subtract_operation.SubtractOperation(a_num, b_num)

        resp = make_response(jsonify(result=op.process()))
        return resp
    except (TypeError, KeyError) as e:
        resp = make_response(jsonify(result=str(e)))
        resp.content_type = 'application/json'
        resp.status_code = 400
        return resp


@app.route('/mul', methods=["POST"])
def multiply_service():
    try:
        request_data = request.json
        a_num = request_data["a"]
        b_num = request_data["b"]
        op = multiply_operation.MultiplyOperation(a_num, b_num)

        resp = make_response(jsonify(result=op.process()))
        return resp
    except (TypeError, KeyError) as e:
        resp = make_response(jsonify(result=str(e)))
        resp.content_type = 'application/json'
        resp.status_code = 400
        return resp


@app.route('/div', methods=["POST"])
def divide_service():
    try:
        request_data = request.json
        a_num = request_data["a"]
        b_num = request_data["b"]
        op = divide_operation.DivideOperation(a_num, b_num)

        resp = make_response(jsonify(result=op.process()))
        return resp
    except (TypeError, KeyError) as e:
        resp = make_response(jsonify(result=str(e)))
        resp.content_type = 'application/json'
        resp.status_code = 400
        return resp


if __name__ == '__main__':
    app.run()
