import tensorflow as tf 
import numpy as np 



class ImageCorpus:
    def __init__(self):
        self.image_width = 5
        self.image_height = 5

        self.learning_rate = 1e-3

        self.forward_pass = self.forward_pass_tensor()

        self.loss = self.loss_tensor()
        self.optimize = self..optimize_tensor()
        

class Model:
    def __init__(self, corpus):
        self.corpus = corpus

        # below currently uses black and white images. We can change if need be later. 
        self.input = tf.placeholder(tf.float32, shape=[None, corpus.image_width, corpus.image_height])

        # These labels will be an integer that represents the park that the given image is associated with
        self.output = tf.placeholder(tf.int32, shape=[None])

    def forward_pass_tensor(self):
        pass

    def loss_tensor(self):
        pass

    def optimize_tensor(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).optimize(self.loss)

    def train(self):
        pass


    def test(self):
        pass