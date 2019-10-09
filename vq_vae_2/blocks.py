"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def downsampling_block(
        inputs,
        hidden_size=128,
        num_downsampling_layers=1,
):
    x = inputs
    for layer in range(num_downsampling_layers):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            hidden_size // (2 ** (num_downsampling_layers - 1 - layer)),
            4,
            strides=(2, 2),
            padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.layers.Conv2D(
        hidden_size,
        3,
        padding='same')(x)


def upsampling_block(
        inputs,
        hidden_size=128,
        num_upsampling_layers=1,
):
    x = inputs
    for layer in range(num_upsampling_layers):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2DTranspose(
            hidden_size // (2 ** (layer + 1)),
            4,
            strides=(2, 2),
            padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.layers.Conv2D(
        hidden_size,
        3,
        padding='same')(x)


def residual_block(
        inputs,
        hidden_size=32,
        output_size=128,
        num_residual_layers=1,
):
    for layer in range(num_residual_layers):
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            hidden_size,
            3,
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            output_size,
            1,
            padding='same')(x)
        inputs = tf.keras.layers.add([inputs, x])
    return inputs
