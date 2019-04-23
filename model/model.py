import tensorflow as tf 
import numpy as np 



class ImageCorpus:
    def __init__(self, width, height, num_labels, image_directory):
        self.image_width = width
        self.image_height = height
        self.num_labels = num_labels

        self.batch_size = 128



        self.input_dataset_iterator, self.output_dataset_iterator = self.load_image_batch(image_directory)



    def get_batch(self):
        """
        Gets a fixed batch size of images, returning a numpy array 
        of [bsz, width, height, 3], where bsz is batch size or some 
        number less than that if necessary for our data. 
        """
        

        return np.zeros(shape=[self.batch_size, self.width, self.height, 3])


    def load_image_batch(dirname):
        shuffle_buffer_size = 250000
        n_threads = 2

        def load_and_process_image(filename):
            image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)

            image = (image - 0.5) * 2

            return image

        def label_image(filename):
            """
            This function will input a filename and output the label 
            for which it belongs
            """
            
            return 0 # TODO fix
        
        dir_path = dirname + '/*.jpg'

        dataset = tf.data.Dataset.list_files(dir_path)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        input_dataset = dataset.map(map_func=load_and_process_image, num_paralle_calls=n_threads)
        output_dataset = dataset.map(map_func=label_image, num_paralle_calls=n_threads)


        input_dataset = input_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size=self.batch_size)
        )
        output_dataset = output_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size=self.batch_size)
        )

        input_dataset = input_dataset.prefetch(1)
        output_dataset = output_dataset.prefetch(1)

        return input_dataset.make_initializable_iterator(), output_dataset.make_initializable_iterator()

class Model:
    def __init__(self, corpus):
        self.corpus = corpus
        self.image_width = corpus.image_width 
        self.image_height = corpus.image_height
        self.num_labels = corpus.num_labels



        self.num_epochs = 5
        self.learning_rate = 1e-3

        self.input = self.corpus.input_dataset_iterator.get_next()

        # These labels will be an integer that represents the park that the given image is associated with
        self.output = tf.placeholder(tf.int32, shape=[None])
        self.output = self.corpus.output_dataset_iterator.get_next()

        self.forward_pass = self.forward_pass_tensor()
        print(self.forward_pass.shape)

        self.loss = self.loss_tensor()
        self.optimize = self.optimize_tensor()


        self.sess = tf.Session() 
        self.sess.run(tf.global_variable_initializer())

    def forward_pass_tensor(self):

        conv = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2,2], strides=2)

        print(pool.shape)

        for i in range(5):

            conv = tf.layers.conv2d(inputs=pool, filters=32, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2,2], strides=2)


        

        flattened = tf.reshape(self.input, [-1, np.prod(pool.shape[1:])])
        return tf.layers.dense(flattened, self.num_labels)
        

    def loss_tensor(self):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input, logits=self.forward_pass)

    def optimize_tensor(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        for epoch in range(self.num_epochs):
            self.sess.run(corpus.dataset_iterator.initializer)

            """
            Do some things here
            """

            sess.run(dataset_iterator.initializer)

        pass



    def test(self):
        pass






if __name__ == '__main__':

    print('Hi, if you are seeing this, welcome to debugging the model!')
    model = Model(ImageCorpus(64, 64, 100, "./"))