import tensorflow as tf
from tensorflow import keras


def get_model(model_params, optim_params):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(model_params.dense_units, activation=model_params.activation),
        keras.layers.Dropout(model_params.dropout),
        keras.layers.Dense(model_params.dense_units, activation=model_params.activation),
        keras.layers.Dropout(model_params.dropout),
        keras.layers.Dense(model_params.dense_units, activation=model_params.activation),
        keras.layers.Dropout(model_params.dropout),
        keras.layers.Dense(10, activation='softmax')
    ])

    if optim_params.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=optim_params.init_learning_rate,
        )
    elif optim_params.optimizer == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=optim_params.init_learning_rate,
        )
    elif optim_params.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=optim_params.init_learning_rate,
        )

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
