"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from vq_vae_2.encoder import encoder
from vq_vae_2.decoder import decoder
from vq_vae_2.vector_quantizer import VectorQuantizer
import tensorflow as tf


def vq_vae_2(
        inputs,
        num_quantizer_layers=2,
        hidden_size=128,
        num_embeddings=512,
        embedding_size=64,
        decay=0.99,
):
    feature_pyramid = [inputs]
    for layer in range(num_quantizer_layers):
        x = encoder(
            feature_pyramid[-1],
            initial_hidden_size=hidden_size // 2 if layer == 0 else hidden_size,
            downsampling_hidden_size=hidden_size,
            residual_hidden_size=hidden_size // 4,
            num_downsampling_layers=2 if layer == 0 else 1,
            num_residual_layers=2)
        feature_pyramid.append(x)
    x = feature_pyramid.pop()
    for layer in reversed(range(num_quantizer_layers)):
        x = tf.keras.layers.Conv2D(
            embedding_size,
            1,
            padding='same')(x)
        x = VectorQuantizer(num_embeddings=num_embeddings, decay=decay)(x)
        x = decoder(
            x,
            initial_hidden_size=hidden_size // 2 if layer == 0 else hidden_size,
            upsampling_hidden_size=hidden_size,
            residual_hidden_size=hidden_size // 4,
            num_upsampling_layers=2 if layer == 0 else 1,
            num_residual_layers=2)
        if layer > 0:
            x = tf.keras.layers.concatenate([x, feature_pyramid.pop()])
    return tf.keras.layers.Conv2D(
        3,
        1,
        padding='same')(x)
