from flask import Flask, current_app, json, request
from gevent.pywsgi import WSGIServer
import numpy as np
import scipy.ndimage
import cv2
import tensorflow as tf
import argparse
import sys
import json


app = Flask(__name__)

session = ""
input_tensor = ""
predict_tensor = ""
model = ""

# class SavedModel:
# 	def __init__(self, input_tensor, predict_tensor, model)

@app.route('/', methods = ['GET'])
def index():
    return current_app.send_static_file('index.html')

@app.route('/upload', methods = ['POST'])
def upload():
	filestr = request.files['file'].read()
	npimg = np.fromstring(filestr, np.uint8)
	image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)


	image = ((image / 255) - 0.5) * 2


	image = cv2.resize(image, dsize=(600, 400), interpolation=cv2.INTER_CUBIC)

	images = np.array([image])
	return json.dumps(predict(image))

def predict(image):
	images = np.array([image])
	results = session.run(predict_tensor,feed_dict={input_tensor:images})

	arr = results[0]
	return format_result_array(arr)

def format_result_array(arr):
	print(arr)
	if len(arr) == 4:
		file = "../regions/park_labels.json"
	else:
		file = "../regions/land_labels.json"
	with open(file, 'r') as lbl_file:
		data = json.load(lbl_file)
		toReturn = {}
		for key in data:
			idx = int(key) - 1
			toReturn[data[key]['name']] = arr[idx]
	print(toReturn)
		return str(toReturn)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run webserver that will predict the park/land of an image')
	parser.add_argument('-model', help='Path to the model name. e.g., "park/model-epoch3-100"')
	parser.parse_args()
	model = sys.argv[2]
	model_path = "../model/" + model
	session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
	print("Recreating graph structure from meta file\n")
	"../model/"
	new_saver = tf.train.import_meta_graph(model_path + ".meta")
	print("Restoring variables from %s\n" % model_path)
	new_saver.restore(session, model_path)
	print("Initializing...")
	session.run(tf.global_variables_initializer())
	print("Getting default graph...")
	graph = tf.get_default_graph()
	print("Getting input tensor")
	input_tensor = graph.get_tensor_by_name("Placeholder:0")
	print("Getting prediction tensor")
	predict_tensor = graph.get_tensor_by_name("dense_2/BiasAdd:0")
	print("Starting webserver")
	http_server = WSGIServer(('', 80), app)
	http_server.serve_forever()