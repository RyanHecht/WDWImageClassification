import tensorflow as tf 
import numpy as np 



class ImageCorpus:
    def __init__(self, width, height, num_labels):
        self.image_width = width
        self.image_height = height
        self.num_labels = num_labels


        

class Model:
    def __init__(self, corpus):
        self.corpus = corpus
        self.image_width = corpus.image_width 
        self.image_height = corpus.image_height
        self.num_labels = corpus.num_labels


        self.learning_rate = 1e-3



        # below currently uses black and white images. We can change if need be later. 
        self.input = tf.placeholder(tf.float32, shape=[None, self.image_width, self.image_height, 3])

        # These labels will be an integer that represents the park that the given image is associated with
        self.output = tf.placeholder(tf.int32, shape=[None])


        self.forward_pass = self.forward_pass_tensor()
        print(self.forward_pass.shape)

        self.loss = self.loss_tensor()
        self.optimize = self.optimize_tensor()

    def forward_pass_tensor(self):

        conv = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2,2], strides=2)

        print(pool.shape)

        

        flattened = tf.reshape(self.input, [-1, np.prod(pool.shape[1:])])
        return tf.layers.dense(flattened, self.num_labels)
        

    def loss_tensor(self):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input, logits=self.forward_pass)

    def optimize_tensor(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        pass


    def test(self):
        pass






if __name__ == '__main__':

    print('Hi, if you are seeing this, welcome to debugging the model!')
    model = Model(ImageCorpus(28, 28, 100))