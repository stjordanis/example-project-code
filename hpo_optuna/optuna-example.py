import hashlib
import os
import tempfile

import matplotlib.pyplot as plt
import neptune
import numpy as np
import optuna
import tensorflow as tf
from tensorflow import keras


def log_data(logs):
    neptune.log_metric('epoch_accuracy', logs['accuracy'])
    neptune.log_metric('epoch_categorical_crossentropy', logs['categorical_crossentropy'])
    neptune.log_metric('epoch_loss', logs['loss'])


def train_evaluate(params):
    def lr_scheduler(epoch):
        if epoch < 10:
            new_lr = params['learning_rate']
        else:
            new_lr = params['learning_rate'] * np.exp(0.1 * ((epoch // 10) * 10 - epoch))

        neptune.log_metric('learning_rate', new_lr)
        return new_lr

    # create experiment
    neptune.create_experiment(name='optuna_example',
                              tags=['optuna'],
                              upload_source_files=['optuna-example.py', 'requirements.txt'],
                              params=params)

    neptune.set_property('train_images_version', hashlib.md5(train_images).hexdigest())
    neptune.set_property('train_labels_version', hashlib.md5(train_labels).hexdigest())
    neptune.set_property('test_images_version', hashlib.md5(test_images).hexdigest())
    neptune.set_property('test_labels_version', hashlib.md5(test_labels).hexdigest())
    neptune.set_property('class_names', class_names)

    for j, class_name in enumerate(class_names):
        plt.figure(figsize=(10, 10))
        label_ = np.where(train_labels == j)
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[label_[0][i]], cmap=plt.cm.binary)
            plt.xlabel(class_names[j])
        neptune.log_image('example_images', plt.gcf())

    # model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(params['hidden_layer_size'], activation=params['activation']),
        keras.layers.Dense(params['hidden_layer_size'], activation=params['activation']),
        keras.layers.Dense(params['hidden_layer_size'], activation=params['activation']),
        keras.layers.Dense(10, activation='softmax')
    ])

    # optimizer
    if params['optimizer'] == 'Adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params['learning_rate'],
            beta_1=params['optimizer_beta_1'],
            beta_2=params['optimizer_beta_2'],
            )
    elif params['optimizer'] == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=params['learning_rate'],
            beta_1=params['optimizer_beta_1'],
            beta_2=params['optimizer_beta_2'],
            )

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'categorical_crossentropy'])

    # log model summary
    model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

    # train model
    model.fit(train_images, train_labels,
              batch_size=params['batch_size'],
              epochs=params['n_epochs'],
              shuffle=params['shuffle'],
              callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_data(logs)),
                         keras.callbacks.EarlyStopping(patience=params['early_stopping'],
                                                       monitor='accuracy',
                                                       restore_best_weights=True),
                         keras.callbacks.LearningRateScheduler(lr_scheduler)]
              )

    # log model weights
    with tempfile.TemporaryDirectory(dir='.') as d:
        prefix = os.path.join(d, 'model_weights')
        model.save_weights(os.path.join(prefix, 'model'))
        for item in os.listdir(prefix):
            neptune.log_artifact(os.path.join(prefix, item),
                                 os.path.join('model_weights', item))

    # evaluate model
    eval_metrics = model.evaluate(test_images, test_labels, verbose=0)
    for j, metric in enumerate(eval_metrics):
        neptune.log_metric('eval_' + model.metrics_names[j], metric)
        if model.metrics_names[j] == 'accuracy':
            score = metric

    neptune.stop()
    return score


def objective(trial):
    optuna_params = {'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
                     'activation': trial.suggest_categorical('activation', ['relu', 'elu']),
                     'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.1),
                     'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'Nadam'],),
                     'hidden_layer_size': trial.suggest_categorical('hidden_layer_size', [16, 32, 64, 128]),
                     }
    PARAMS = {**optuna_params, **STATIC_PARAMS}
    return train_evaluate(PARAMS)


# static params
STATIC_PARAMS = {'n_epochs': 100,
                 'shuffle': True,
                 'optimizer_beta_1': 0.9,
                 'optimizer_beta_2': 0.999,
                 'early_stopping': 10,
                 }

# dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# select project
neptune.init('kamil/sandbox')

# make optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# run experiment that collects study visuals
neptune.create_experiment(name='optuna_summary',
                          tags=['optuna', 'optuna-summary'],
                          upload_source_files=['optuna-example.py', 'requirements.txt'])
neptune.log_metric('best_score', study.best_value)
neptune.set_property('best_parameters', study.best_params)
neptune.stop()
