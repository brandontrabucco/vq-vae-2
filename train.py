"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from vq_vae_2.vq_vae_2 import vq_vae_2
import tensorflow as tf


if __name__ == "__main__":

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
    model = tf.keras.models.Model(input_layer, vq_vae_2(input_layer))
    optimizer = tf.keras.optimizers.Adam(lr=0.0003)

    tf.io.gfile.makedirs("./")
    writer = tf.summary.create_file_writer("./")

    train_data, test_data = tf.keras.datasets.cifar10.load_data()
    train_features, train_labels = train_data
    test_features, test_labels = test_data

    train_dataset = tf.data.Dataset.from_tensor_slices({
        "images": train_features, "labels": train_features})
    val_dataset = tf.data.Dataset.from_tensor_slices({
        "images": test_features, "labels": test_features})

    def normalize_image(batch):
        return {
            "images": (tf.cast(batch["images"], tf.float32) / 255.0) - 0.5,
            "labels": tf.cast(batch["labels"], tf.int32)}

    train_dataset = train_dataset.map(
        normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(10000)
    train_dataset = train_dataset.repeat(-1)
    train_dataset = train_dataset.batch(128)

    val_dataset = val_dataset.map(
        normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.shuffle(1000)
    val_dataset = val_dataset.repeat(-1)
    val_dataset = val_dataset.batch(128)

    dataset = tf.data.Dataset.zip({"train": train_dataset, "val": val_dataset})
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    for iteration, batch in enumerate(dataset):

        with tf.GradientTape() as tape:
            train_prediction = model(batch["train"]["images"], training=True)
            train_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                batch["train"]["images"], train_prediction))

        optimizer.apply_gradients(zip(tape.gradient(
            train_loss, model.trainable_variables), model.trainable_variables))

        tf.summary.experimental.set_step(iteration)
        print("Iteration: {} MSE: {}".format(iteration, train_loss.numpy()))
        with writer.as_default():
            tf.summary.image("original", batch["train"]["images"] + 0.5)
            tf.summary.image("reconstruction", train_prediction + 0.5)
            tf.summary.scalar("mse", train_loss)
