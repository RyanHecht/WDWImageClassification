from flask import Flask, current_app
from gevent.pywsgi import WSGIServer
app = Flask(__name__)


@app.route('/', methods = ['GET'])
def index():
    return current_app.send_static_file('index.html')

@app.route('/upload', methods = ['POST'])
def upload():
    return {"test": "555"}

if __name__ == "__main__":
	print("Starting webserver")
	http_server = WSGIServer(('', 8080), app)
	http_server.serve_forever()