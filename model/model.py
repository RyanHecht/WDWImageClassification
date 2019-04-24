import tensorflow as tf 
import numpy as np 

import os 
import random

import json
import scipy.ndimage


class ImageCorpus:
    def __init__(self, width, height, num_labels, image_directory):
        self.image_width = width
        self.image_height = height
        self.num_labels = num_labels

        self.batch_size = 120

        self.batch_from_file()
        self.input_dataset_iterator, self.output_dataset_iterator = self.load_image_batch(image_directory)





    def load_image_batch(self, dirname):
        shuffle_buffer_size = 250000
        n_threads = 2

        def load_and_process_image(filename):
            image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)

            image = (image - 0.5) * 2

            image = image[0:64, 0:64]


            

            return image

        def label_image(filename):
            """
            This function will input a filename and output the label 
            for which it belongs
            """
            ident = tf.strings.substr(filename, 0, 7)
            json = 'labels/' + ident + ".json"
            content_json = tf.io.read_file(json)
            tf.io.decode_json_example(content_json)


            return 0 #np.array([0]) # TODO fix
        
        dir_path = dirname + '/*.jpg'

        dataset = tf.data.Dataset.list_files(dir_path)
        
        print(dataset)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        input_dataset = dataset.map(map_func=load_and_process_image, num_parallel_calls=n_threads)
        output_dataset = dataset.map(map_func=label_image, num_parallel_calls=n_threads)


        input_dataset = input_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size=self.batch_size)
        )
        output_dataset = output_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size=self.batch_size)
        )

        input_dataset = input_dataset.prefetch(1)
        output_dataset = output_dataset.prefetch(1)



        return input_dataset.make_initializable_iterator(), output_dataset.make_initializable_iterator()




    def batch_from_file(self):
        json_filenames = random.choices(os.listdir('./data/labels'), k =self.batch_size)
        numbers = [st[:-5] for st in json_filenames]
        
        image_filenames = [n + '.jpg' for n in numbers]


        images = [] 
        labels = []

        for image_name, json_name in zip(image_filenames, json_filenames):

            image = tf.image.decode_jpeg(tf.read_file('./data/' + image_name), channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)

            image = scipy.ndimage.imread('./data/' + image_name, mode='RGB')

            image = (image - 0.5) * 2

            image = image[0:64, 0:64, :]
            
            images.append(image)

            with open('./data/labels/' + json_name) as json_file:
                data = json.load(json_file)
                label = int(data['labels']['park'])
                labels.append(label + 1)



        return np.array(images), np.array(labels)
            
            


        



class Model:
    def __init__(self, corpus):
        self.corpus = corpus
        self.image_width = corpus.image_width 
        self.image_height = corpus.image_height
        self.num_labels = corpus.num_labels



        self.num_epochs = 100
        self.learning_rate = 1e-3

        self.input = tf.placeholder(tf.float32, shape=[None, self.image_width, self.image_height, 3])
        # self.input = self.corpus.input_dataset_iterator.get_next()

        # These labels will be an integer that represents the park that the given image is associated with
        self.output = tf.placeholder(tf.int32, shape=[None])
        # self.output = self.corpus.output_dataset_iterator.get_next()

        self.forward_pass = self.forward_pass_tensor()

        self.loss = self.loss_tensor()
        self.optimize = self.optimize_tensor()

        self.accuracy = self.accuracy_tensor()
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer())

    def forward_pass_tensor(self):

        conv = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2,2], strides=2)


        for i in range(5):

            conv = tf.layers.conv2d(inputs=pool, filters=32, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2,2], strides=2)


        

        flattened = tf.reshape(self.input, [-1, self.image_width * self.image_height * 3])
        return tf.layers.dense(flattened, self.num_labels)
        

    def loss_tensor(self):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output, logits=self.forward_pass)

    def optimize_tensor(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def accuracy_tensor(self):
        """
        Calculates the model's prediction accuracy by comparing
        predictions to correct labels – no need to modify this

        :return: the accuracy of the model as a tensor
        """
        correct_prediction = tf.equal(tf.cast(self.output, tf.int64),
                                      tf.argmax(self.forward_pass, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def train(self):
        for epoch in range(self.num_epochs):
            print("STARTING EPOCH #", epoch)
            #self.sess.run(self.corpus.input_dataset_iterator.initializer)
            #self.sess.run(self.corpus.output_dataset_iterator.initializer)

            """
            Do some things here
            """
            """
            try:
                while True: 
                    l = self.sess.run([self.optimize, self.accuracy])
                    print('acc: ', l)

            except tf.errors.OutOfRangeError:
                pass

            self.sess.run(self.corpus.input_dataset_iterator.initializer)
            self.sess.run(self.corpus.output_dataset_iterator.initializer)
            """


            for i in range(100):
                i, o = self.corpus.batch_from_file()

                l = self.sess.run([self.optimize, self.accuracy], feed_dict = {self.input: i, self.output: o})
                print(l)

        pass



    def test(self):
        pass






if __name__ == '__main__':

    print('Hi, if you are seeing this, welcome to debugging the model!')
    model = Model(ImageCorpus(64, 64, 10, "./data"))
    model.train()