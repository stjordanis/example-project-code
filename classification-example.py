import hashlib
import os
import tempfile

import matplotlib.pyplot as plt
import neptune
import numpy as np
import tensorflow as tf
from tensorflow import keras


def log_data(logs):
    neptune.log_metric('epoch_accuracy', logs['accuracy'])
    neptune.log_metric('epoch_loss', logs['loss'])


def lr_scheduler(epoch):
    if epoch < 10:
        new_lr = PARAMS['learning_rate']
    else:
        new_lr = PARAMS['learning_rate'] * np.exp(0.1 * ((epoch//10)*10 - epoch))

    neptune.log_metric('learning_rate', new_lr)
    return new_lr


# select project
neptune.init('USERNAME/example-project')

# define parameters
PARAMS = {'batch_size': 32,
          'n_epochs': 100,
          'shuffle': True,
          'activation': 'relu',
          'learning_rate': 0.0015,
          'optimizer_beta_1': 0.9,
          'optimizer_beta_2': 0.999,
          'early_stopping': 10,
          'optimizer': 'Adam',
          }

# create experiment
with neptune.create_experiment(name='classification_example',
                               tags=['classification', 'tf-2'],
                               upload_source_files=['classification-example.py', 'requirements.txt'],
                               description='scheduler cyclic, new setup',
                               params=PARAMS):
    # dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    neptune.set_property('train_images_version', hashlib.md5(train_images).hexdigest())
    neptune.set_property('train_labels_version', hashlib.md5(train_labels).hexdigest())
    neptune.set_property('test_images_version', hashlib.md5(test_images).hexdigest())
    neptune.set_property('test_labels_version', hashlib.md5(test_labels).hexdigest())

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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
        keras.layers.Dense(128, activation=PARAMS['activation']),
        keras.layers.Dense(128, activation=PARAMS['activation']),
        keras.layers.Dense(128, activation=PARAMS['activation']),
        keras.layers.Dense(10, activation='softmax')
    ])

    if PARAMS['optimizer'] == 'Adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=PARAMS['learning_rate'],
            beta_1=PARAMS['optimizer_beta_1'],
            beta_2=PARAMS['optimizer_beta_2'],
        )
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=PARAMS['learning_rate'],
        )

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # log model summary
    model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

    # train model
    model.fit(train_images, train_labels,
              batch_size=PARAMS['batch_size'],
              epochs=PARAMS['n_epochs'],
              shuffle=PARAMS['shuffle'],
              callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_data(logs)),
                         keras.callbacks.EarlyStopping(patience=PARAMS['early_stopping'],
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