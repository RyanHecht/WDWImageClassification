import tensorflow as tf 
import numpy as np 



class ImageCorpus:
    def __init__(self):
        self.image_wnameth = 5
        self.image_height = 5


        

class Model:
    def __init__(self, corpus):
        self.corpus = corpus
        self.image_wnameth = corpus.image_wnameth 
        self.image_height = corpus.image_height


        self.learning_rate = 1e-3



        # below currently uses black and white images. We can change if need be later. 
        self.input = tf.placeholder(tf.float32, shape=[None, self.image_wnameth, self.image_height])

        # These labels will be an integer that represents the park that the given image is associated with
        self.output = tf.placeholder(tf.int32, shape=[None])


        self.forward_pass = self.forward_pass_tensor()
        print(self.forward_pass.shape)

        self.loss = self.loss_tensor()
        self.optimize = self.optimize_tensor()

    def forward_pass_tensor(self):
        flattened = tf.reshape(self.input, [-1, self.image_wnameth * self.image_height])
        return tf.layers.dense(flattened, 1)
        

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
    model = Model(ImageCorpus())