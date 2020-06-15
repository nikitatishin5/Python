
import time
import bcrypt
from flask import request, abort, Flask, session
from flask.json import jsonify
import json
import uuid
import datetime
import pytest

import requests



session_file = "session.json"
file = "user.txt"

app = Flask(__name__)

@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app .test_client() as client:
        yield  client

def test_client(client):
    rv = client.get('/users')
    assert b'No end' in rv.data


def mock_pred(s: str)->str:
    return "None"

def write_session(user_data: dict):
    last = read_session()
    f = open(session_file, 'w')
    last.append(user_data)
    f.write(json.dumps(last))
    f.close()


def read_session() -> dict:
    f = open(session_file,'r')
    data = json.load(f)
    f.close()
    print(type(data))
    print(data)
    return data

def change_name(name,new_name):
    data = read_session()
    for i in data:
        if i['name'] == name:
            i['name'] = new_name
            break
    f = open(session_file, 'w')
    f.write(json.dumps(data))
    f.close()

@app.route("/users")
def get():
    d = read_session()
    s = ' '
    for i in d:
        s = s + str(i['name']) + ' '
    return s

@app.route("/auth", methods = ['POST'])
def get_hash():
    if not request.json or not 'name' in request.json or not 'pass' in request.json:

        abort(400)

    data = read_session()
    input_pass = request.json['pass']
    for i in data:
        hash_password = i['pass'].encode(errors = 'surrogateescape')
        if bcrypt.checkpw(input_pass.encode(errors = 'surrogateescape'),hash_password):
            f = open(file, 'w')

            uniq_str = uuid.uuid4().hex[:6].upper()
            f.write(uniq_str)
            f.close()
            return {"ok":uniq_str}

        else:
            return {"error":"not authorized"}

@app.route("/user/<name>", methods = ['POST'])
def get_user(name):


    in_str = request.headers['auth']
    print(in_str)
    f = open(file, 'r')
    un = f.read()
    print(un)
    if in_str == un:
        change_name(name,request.json['name'])
        return "ok"
    else:
        return {"error":"not authorized"}


    # data = read_session()
    # input_pass = request.json['pass']
    # for i in data:
    #     hash_password = i['pass'].encode(errors = 'surrogateescape')
    #     if bcrypt.checkpw(input_pass.encode(errors = 'surrogateescape'),hash_password):
    #         uniq_str = uuid.uuid4().hex[:6].upper()
    #         return {"ok":uniq_str}
    #     else:
    #         return {"error":"not authorized"}
    return name

@app.route("/test", methods = ['POST'])
def hh():
    return jsonify(request.headers['auth'])


def mock_pred(s: str)->str:
    return "None"

def mock_Upper(s: str) -> str:
    return s.upper()

# @api.route("/<name>_<passw>")
# class Predict(Resource):
#     def get(self, name, passw):
#         t = time.time()
#         cr_pass = bcrypt.hashpw(str.encode(passw),bcrypt.gensalt())
#         print(cr_pass)
#         write_session({"name":name, "pass":str(cr_pass), "time":t})
#         return mock_Upper("predict done!")

@app.route('/user', methods = ['POST'])
def create_task():
    if not request.json or not 'name' in request.json or not 'pass' in request.json:

        abort(400)

    t = time.time()
    print(request.json['pass'])
    print(type(request.json['pass']))
    print(str.encode(request.json['pass']))
    print(type(str.encode(request.json['pass'])))
    cr_pass = bcrypt.hashpw(str.encode(request.json['pass']),bcrypt.gensalt())
    # print(request.json)
    # print(cr_pass)
    write_session({"name":request.json['name'], "pass":cr_pass.decode(errors = 'surrogateescape'), "time":t})
    return mock_Upper("user done!")

if __name__ == "__main__":
    app.secret_key = "key"
    app.permanent_session_lifetime = datetime.timedelta(days=365)
    app.run()




