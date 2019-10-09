"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


class VectorQuantizer(tf.keras.layers.Layer):

    def __init__(
            self,
            num_embeddings=512,
            decay=0.99,
            **kwargs
    ):
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.ema_cluster_size = None
        self.ema_dw = None
        self.embeddings = None
        super(VectorQuantizer, self).__init__(**kwargs)

    def get_config(
            self
    ):
        base_config = super(VectorQuantizer, self).get_config()
        return dict(num_embeddings=self.num_embeddings, decay=self.decay, **base_config)

    def compute_output_shape(
            self,
            input_shape
    ):
        return input_shape

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(input_shape[-1], self.num_embeddings),
            initializer=tf.keras.initializers.VarianceScaling(distribution='uniform'),
            trainable=False)
        self.ema_dw = self.add_weight(
            name='ema_dw',
            shape=(input_shape[-1], self.num_embeddings),
            initializer=tf.keras.initializers.Constant(value=self.embeddings.numpy()),
            trainable=False)
        self.ema_cluster_size = self.add_weight(
            name='ema_cluster_size',
            shape=(self.num_embeddings,),
            initializer=tf.keras.initializers.zeros,
            trainable=False)
        super(VectorQuantizer, self).build(input_shape)

    def update_embeddings(
            self,
            flat_inputs,
            encodings
    ):
        self.ema_cluster_size.assign(
            tf.keras.backend.moving_average_update(
                self.ema_cluster_size, tf.reduce_sum(encodings, axis=0), self.decay))
        self.ema_dw.assign(
            tf.keras.backend.moving_average_update(
                self.ema_dw, tf.matmul(flat_inputs, encodings, transpose_a=True), self.decay))
        n = tf.reduce_sum(self.ema_cluster_size)
        self.embeddings.assign(
            self.ema_dw / tf.reshape(n * (self.ema_cluster_size + 1e-5) / (
                n + self.num_embeddings * 1e-5), [1, -1]))

    def call(
            self,
            inputs,
            training=False
    ):
        flat_inputs = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
        distances = (
            tf.reduce_sum(flat_inputs ** 2, 1, keepdims=True) -
            2 * tf.matmul(flat_inputs, self.embeddings) +
            tf.reduce_sum(self.embeddings ** 2, 0, keepdims=True))
        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        if training:
            self.update_embeddings(flat_inputs, encodings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = tf.nn.embedding_lookup(
            tf.transpose(self.embeddings, [1, 0]), encoding_indices)
        return inputs + tf.stop_gradient(quantized - inputs)
