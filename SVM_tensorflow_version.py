import numpy as np
import tensorflow as tf
from numpy import vectorize
from sklearn.metrics import f1_score


class SVMClassifier:

    def __init__(self, train_data=None):
        data, labels = train_data

        labels = self._transform_labels(labels)
        data = self._flatten_input(data)

        self.train_data = (data, labels)

        self.assemble_graph()

        self._open_session()

        if train_data:
            self.train()

    def assemble_graph(self, learning_rate=0.02):
        n_features = self.train_data[0].shape[1]
        initializer = tf.contrib.layers.xavier_initializer()
        self.x = tf.placeholder(shape=(None, n_features), dtype=np.float32)
        self.y = tf.placeholder(shape=(None, 1), dtype=np.float32)
        self.w = tf.get_variable(shape=(n_features, 1), initializer=initializer, name='W')
        self.b = tf.get_variable(shape=(), initializer=initializer, name='b')
        self.loss = tf.nn.relu(1 - tf.multiply(self.y, tf.matmul(self.x, self.w) - self.b))
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def train(self, epochs=20, minibatch_size=256):
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for batch in self._create_minibatches(minibatch_size):
                _, loss = self.sess.run([self.train_op, self.loss],
                                        feed_dict={self.x: batch[0], self.y: batch[1]})
    def predict(self, data):
        x = tf.cast(self._flatten_input(data), tf.float32)
        pred = tf.clip_by_value(tf.matmul(x, self.w) - self.b, 0, 1)
        return self.sess.run(pred)

    def _create_minibatches(self, minibatch_size):
        pos = 0

        data, labels = self.train_data
        n_samples = len(labels)

        batches = []
        while pos + minibatch_size < n_samples:
            batches.append((data[pos:pos + minibatch_size, :], labels[pos:pos + minibatch_size]))
            pos += minibatch_size

        if pos < n_samples:
            batches.append((data[pos:n_samples, :], labels[pos:n_samples, :]))

        return batches

    def _transform_labels(self, labels):
        return labels.reshape(-1, 1)

    def _flatten_input(self, data):
        shape = data.shape
        return np.reshape(data, newshape=(shape[0], shape[1] * shape[2]))

    def _open_session(self):
        self.sess = tf.Session()


if __name__ == "__main__":

    def mnist_to_binary(train_data, train_label, test_data, test_label):

        binarized_labels = []
        remainder_2 = vectorize(lambda x: x % 2)
        for labels in [train_label, test_label]:
            binarized_labels.append(remainder_2(labels))

        train_label, test_label = binarized_labels

        return train_data, train_label, test_data, test_label


    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data, train_labels, test_data, test_labels = mnist_to_binary(train_data, train_labels, eval_data, eval_labels)

    svm = SVMClassifier((train_data, train_labels))
    print("Testing score f1: {}".format(f1_score(test_labels, svm.predict(test_data))))
