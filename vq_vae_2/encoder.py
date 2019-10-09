"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from vq_vae_2.blocks import downsampling_block, residual_block
import tensorflow as tf


def encoder(
        inputs,
        initial_hidden_size=64,
        downsampling_hidden_size=128,
        residual_hidden_size=32,
        num_downsampling_layers=1,
        num_residual_layers=1,
):
    x = tf.keras.layers.Conv2D(
        initial_hidden_size,
        1,
        padding='same')(inputs)
    x = downsampling_block(
        x,
        hidden_size=downsampling_hidden_size,
        num_downsampling_layers=num_downsampling_layers)
    return residual_block(
        x,
        hidden_size=residual_hidden_size,
        output_size=downsampling_hidden_size,
        num_residual_layers=num_residual_layers)
