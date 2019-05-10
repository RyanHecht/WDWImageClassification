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
TEST_SIZE = 0.10


def transform_location(lat, lng):
    final_lat = (lat - 28.38895) * 1000
    final_lng = (-81.5583 - lng) * 1000

    return (final_lat, final_lng)

class ImageCorpus:
    def __init__(self, width, height, image_directory):
        self.image_width = width
        self.image_height = height

        self.batch_size = 1
        # create 90-10 test split of data
        self.file_names = np.array([x for x in os.listdir('./data/labels') if '_' in x])
        test_inds = np.random.choice(range(len(self.file_names)), size = int(len(self.file_names) * TEST_SIZE), replace = False)

        self.test_file_names = self.file_names[test_inds]
        self.file_names = np.delete(self.file_names, test_inds).tolist()
        


    def batch_from_file(self, train_bool=True):
        if train_bool:
            json_filenames = np.random.choice(self.file_names, size=self.batch_size)
        else:
            json_filenames = np.random.choice(self.test_file_names, size=self.batch_size)
        numbers = [st[:-5] for st in json_filenames]
        
        image_filenames = [n + '.jpg' for n in numbers]



        images = [] 
        labels = []

        for image_name, json_name in zip(image_filenames, json_filenames):

            image = tf.image.decode_jpeg(tf.read_file('./data/' + image_name), channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)

            image = scipy.ndimage.imread('./data/' + image_name, mode='RGB')

            image = ((image / 255) - 0.5) * 2


            image = cv2.resize(image, dsize=(600, 400), interpolation=cv2.INTER_CUBIC)
            # image = image[0:64, 0:64, :] 
            

            with open('./data/labels/' + json_name) as json_file:
                data = json.load(json_file)
                loc = transform_location(float(data['location']['lat']), float(data['location']['lng']))

                labels.append(loc)

            
            images.append(image)

        return np.array(images), np.array(labels)
    


class Model:
    def __init__(self, corpus):
        self.corpus = corpus
        self.image_width = corpus.image_width 
        self.image_height = corpus.image_height



        self.num_epochs = 100000000000
        self.learning_rate = 0.0001

        self.input = tf.placeholder(tf.float32, shape=[None, self.image_height, self.image_width, 3], name="input")
        print("input")
        print(self.input)
        # self.input = self.corpus.input_dataset_iterator.get_next()

        # These labels will be an integer that represents the park that the given image is associated with
        self.output = tf.placeholder(tf.float32, shape=[None, 2], name="output")
        self.training = tf.placeholder(dtype=tf.bool, name="training")
        # self.output = self.corpus.output_dataset_iterator.get_next()

        self.forward_pass = self.forward_pass_tensor()
        print("forward")
        print(self.forward_pass)
        self.loss = self.loss_tensor()
        self.optimize = self.optimize_tensor()

        self.saver = tf.train.Saver()
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer())

    def forward_pass_tensor(self):

        conv = tf.layers.conv2d(inputs=self.input, filters=96, strides=(4,4), kernel_size=[11, 11], padding='same', activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[3,3], strides=2)

        conv = tf.layers.conv2d(inputs=pool, filters=256, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[3,3], strides=2)

        conv = tf.layers.conv2d(inputs=pool, filters=384, kernel_size=[3, 3], strides=2, padding='same', activation=tf.nn.relu)
        conv = tf.layers.conv2d(inputs=conv, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)

        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[3,3], strides=4)
        """
        for i in range(4):

            conv = tf.layers.conv2d(inputs=pool, filters=2, kernel_size=[7,7], padding='same', activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2,2], strides=2)



        for i in range(8):

            conv = tf.layers.conv2d(inputs=pool, filters=2, kernel_size=[7,7], padding='same', activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2,2], strides=1)

        """
        print(pool.shape)
        dist = np.prod(pool.shape[1:])

        
        #print("\n\n\nDIASGFL ASIGFJ AOSFIJ ASOFI JASFOA JISFAOS FASF \n\n\n", dist)

        flattened = tf.reshape(pool, [-1, dist])
        flattened = tf.layers.dense(flattened, 1000)
        flattened = tf.layers.dropout(flattened, rate=0.5, training=self.training)
        flattened = tf.layers.dense(flattened, 1000)
        flattened = tf.layers.dropout(flattened, rate=0.5, training=self.training)


        return tf.layers.dense(flattened, 2, name="forwardpass")

        

    def loss_tensor(self):
        return tf.reduce_mean(tf.norm(self.forward_pass - self.output, axis=1), name="loss")

    def optimize_tensor(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)





    def train(self):
        for epoch in range(self.num_epochs):
            print("STARTING EPOCH #", epoch)

            for i in range(200):
                in_, out_ = self.corpus.batch_from_file()
                print("out: ", out_)
                o, l, fp = self.sess.run([self.optimize, self.loss, self.forward_pass], feed_dict = {self.input: in_, self.output: out_, self.training: True})
                print("forward pass:", fp)
                with open("logggp.txt", "a+") as f:
                    f.write(str(l))
                    f.write("\n")
                
                print(l)
                if i % 100 == 0:
                    
                    ckpt_str = 'model/geo/model-epoch%d' % epoch
                    self.saver.save(self.sess, ckpt_str, global_step = i)


            i, o = self.corpus.batch_from_file(False)
            l = self.sess.run(self.loss, feed_dict = {self.input: i, self.output: o, self.training: False})
            with open("logs/accuracy_geo.txt", "a+") as f:
                f.write("epoch: %d , test_acc %f\n" % (epoch, l))

            #print("\n\nTest loss on epoch %d : %f\n\n" % (epoch, l[0])) 
           





if __name__ == '__main__':

    print('Hi, if you are seeing this, welcome to debugging the geomodel!')
    
    model = Model(ImageCorpus(600, 400, "./data"))

    model.train()
