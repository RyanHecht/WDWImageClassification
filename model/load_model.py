import tensorflow as tf 
import numpy as np 

import os
import sys
import random

import json
import scipy.ndimage
import cv2
import argparse
import pdb 
from model import Model, ImageCorpus



def main():
    
    sess = tf.Session()
    print("Recreating graph structure from meta file\n")
    new_saver = tf.train.import_meta_graph("./park/model-epoch3-100.meta")
    print("Restoring variables from %s\n" % "./park/model-epoch3-100")
    new_saver.restore(sess, "./park/model-epoch3-100")
    print("Initializing...")
    sess.run(tf.global_variables_initializer())
    print("Getting default graph...")
    graph = tf.get_default_graph()
    #print("All tensors: %s\n" % [n.name for n in tf.get_default_graph().as_graph_def().node])
    #print("Finished printing")
    
    #print([t.name for op in graph.get_operations() for t in op.values() if t.name.find("ReadFile") == -1 and t.name.find("convert_image") == -1 and t.name.find("DecodeJpeg") == -1])

    print("Initializing image...")
    

    image = scipy.ndimage.imread('/home/rhecht/Downloads/mainst.jpg', mode='RGB')

    image = ((image / 255) - 0.5) * 2


    image = cv2.resize(image, dsize=(600, 400), interpolation=cv2.INTER_CUBIC)

    images = np.array([image])

    print("getting input tensor")
    input = graph.get_tensor_by_name("Placeholder:0")
    print("getting prediction tensor")
    prediction=graph.get_tensor_by_name("dense_2/BiasAdd:0")
          #newdata=put your data here
    print("running prediction\n\n")
    print(sess.run(prediction,feed_dict={input:images}))

if __name__ == "__main__":
    main()
