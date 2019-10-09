"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from vq_vae_2.blocks import upsampling_block, residual_block
import tensorflow as tf


def decoder(
        inputs,
        initial_hidden_size=128,
        upsampling_hidden_size=128,
        residual_hidden_size=32,
        num_upsampling_layers=1,
        num_residual_layers=1,
):
    x = tf.keras.layers.Conv2D(
        initial_hidden_size,
        1,
        padding='same')(inputs)
    x = residual_block(
        x,
        hidden_size=residual_hidden_size,
        output_size=initial_hidden_size,
        num_residual_layers=num_residual_layers)
    return upsampling_block(
        x,
        hidden_size=upsampling_hidden_size,
        num_upsampling_layers=num_upsampling_layers)
