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

park_model = None
land_model = None
geo_model = None


class SavedModel:
	def __init__(self, model_path, name):
		self.graph = tf.Graph()
		#self.session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}), graph=self.graph)
		self.session = tf.Session(graph=self.graph)
		with self.graph.as_default():
			print(name + ": Recreating graph structure from meta file\n")
			saver = tf.train.import_meta_graph(model_path + ".meta")
			print(name + ": Restoring variables from %s\n" % model_path)
			saver.restore(self.session, model_path)
			print(name + ": Initializing global variables")
			self.session.run(tf.global_variables_initializer())
			print(name + ": Getting tensors")
			self.input_tensor = self.graph.get_tensor_by_name("Placeholder:0")
			self.predict_tensor = tf.nn.softmax(self.graph.get_tensor_by_name("dense_2/BiasAdd:0"))
			self.dropout_tensor = self.graph.get_tensor_by_name("Placeholder_2:0")
	
	def predict(self, image):
		images = np.array([image])
		results = self.session.run(self.predict_tensor,feed_dict={self.input_tensor:images, self.dropout_tensor:False})
		arr = results[0]

		return self.format_result_array(arr)

	def format_result_array(self, arr):
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
				toReturn[data[key]['name']] = float(arr[idx])
		print(toReturn)
		return toReturn

class SavedGeoModel:
	def __init__(self, model_path, name):
		self.graph = tf.Graph()
		#self.session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}), graph=self.graph)
		self.session = tf.Session(graph=self.graph)
		with self.graph.as_default():
			print(name + ": Recreating graph structure from meta file\n")
			saver = tf.train.import_meta_graph(model_path + ".meta")
			print(name + ": Restoring variables from %s\n" % model_path)
			saver.restore(self.session, model_path)
			print(name + ": Initializing global variables")
			self.session.run(tf.global_variables_initializer())
			print(name + ": Getting tensors")
			self.input_tensor = self.graph.get_tensor_by_name("input:0")
			self.predict_tensor = self.graph.get_tensor_by_name("forwardpass/BiasAdd:0")
			self.dropout_tensor = self.graph.get_tensor_by_name("training:0")
	
	def predict(self, image):
		images = np.array([image])
		results = self.session.run(self.predict_tensor,feed_dict={self.input_tensor:images, self.dropout_tensor:False})
		
		print(results)
		result = results[0]
		
		loc = self.untransform_location(result[0], result[1])
		return {"lat": loc[0], "lng": loc[1]}

	def untransform_location(self, lat, lng):
		return ((lat / 1000.0) + 28.38895, -1 * ((lng / 1000.0) + 81.5583))

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

	return json.dumps({'park': park_model.predict(image), 'land': land_model.predict(image), 'geo': geo_model.predict(image)})



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run webserver that will predict the park/land of an image')
	parser.add_argument('-park', help='Path to the park model name. e.g., "park/model-epoch3-100"', required=True)
	parser.add_argument('-land', help='Path to the land model name. e.g., "land/model-epoch3-100"', required=True)
	parser.add_argument('-geo', help='Path to the geo model name. e.g., "geo/model-epoch3-100"', required=True)
	args = parser.parse_args()
	
	park_model_path = "../model/" + args.park
	land_model_path = "../model/" + args.land
	geo_model_path = "../model/" + args.geo

	print(park_model_path)
	print(land_model_path)
	print(geo_model_path)

	geo_model = SavedGeoModel(geo_model_path, "geo")
	park_model = SavedModel(park_model_path, "park")
	land_model = SavedModel(land_model_path, "land")
	
	print("Starting webserver")
	http_server = WSGIServer(('', 80), app)
	http_server.serve_forever()