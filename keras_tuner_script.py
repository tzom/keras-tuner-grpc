import kerastuner as kt
import tensorflow as tf
import tensorflow_datasets as tfds


def build_model(hp):
    """Builds a convolutional model."""
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = inputs
    for i in range(hp.Int('conv_layers', 1, 3, default=3)):
        x = tf.keras.layers.Conv2D(
            filters=hp.Int('filters_' + str(i), 4, 32, step=4, default=8),
            kernel_size=hp.Int('kernel_size_' + str(i), 3, 5),
            activation='relu',
            padding='same')(x)

        if hp.Choice('pooling' + str(i), ['max', 'avg']) == 'max':
            x = tf.keras.layers.MaxPooling2D()(x)
        else:
            x = tf.keras.layers.AveragePooling2D()(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    if hp.Choice('global_pooling', ['max', 'avg']) == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def convert_dataset(item):
    """Puts the mnist dataset in the format Keras expects, (features, labels)."""
    image = item['image']
    label = item['label']
    image = tf.dtypes.cast(image, 'float32') / 255.
    return image, label


def main():
    """Runs the hyperparameter search."""
    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective='val_accuracy',
        max_epochs=8,
        factor=2,
        hyperband_iterations=3,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        directory='results_dir',
        project_name='mnist')

    mnist_data = tfds.load('mnist')
    mnist_train, mnist_test = mnist_data['train'], mnist_data['test']
    mnist_train = mnist_train.map(convert_dataset).shuffle(1000).batch(100).repeat()
    mnist_test = mnist_test.map(convert_dataset).batch(100)

    tuner.search(mnist_train,
                 steps_per_epoch=600,
                 validation_data=mnist_test,
                 validation_steps=100,
                 epochs=20,
                 callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy')])


if __name__ == '__main__':
    main()