import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model


def log_epoch_data(logs, npt):
    npt.log_metric('epoch/accuracy', logs['accuracy'])
    npt.log_metric('epoch/loss', logs['loss'])


def log_visualized_model(model, npt):
    with tempfile.TemporaryDirectory(dir='.') as d:
        img_name = os.path.join(d, 'model_visualization.png')
        plot_model(model, to_file=img_name, show_shapes=True, expand_nested=True)
        npt.log_image('model_visualization',
                      img_name,
                      description='model visualization as produced by keras plot model method')


def log_model_weights(model, epoch, logs, npt):
    with tempfile.TemporaryDirectory(dir='.') as d:
        dir_name = 'model_checkpoints'
        model_dir = 'epoch-{}--acc-{:0.3f}'.format(epoch, logs['accuracy'])
        model.save_weights(os.path.join(d, 'weights'))
        for item in os.listdir(d):
            npt.log_artifact(os.path.join(d, item),
                             os.path.join(dir_name, model_dir, item))


def log_images_sample(class_names, train_labels, train_images, exp):
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
        exp.log_image('example_images', plt.gcf())
        plt.close('all')


def lr_scheduler(epoch, lr, npt):
    if epoch < 20:
        new_lr = lr
    else:
        new_lr = lr * np.exp(0.01 * (20 - epoch))
    npt.log_metric('learning_rate', new_lr)
    return new_lr
