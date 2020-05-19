import hashlib
from functools import partial

import neptune
import tensorflow as tf
import yaml
from attrdict import AttrDict

from src.ml_utils import log_model_weights, log_epoch_data, lr_scheduler, log_images_sample, \
    log_visualized_model
from src.model import get_model

# Select project
neptune.init('USERNAME/example-project')

# Prepare params
with open('src/parameters.yaml', 'r') as f:
    all_params = AttrDict(yaml.safe_load(f))
    model_params = all_params.model
    training_params = all_params.training
    optim_params = all_params.optim_params

# Create experiment
exp = neptune.create_experiment(name='another_classification',
                                tags=['keras'],
                                upload_source_files=['**/*.py', 'parameters.yaml'],
                                params=model_params + training_params + optim_params)

# Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

exp.set_property('train_images_version', hashlib.md5(train_images).hexdigest())
exp.set_property('train_labels_version', hashlib.md5(train_labels).hexdigest())
exp.set_property('test_images_version', hashlib.md5(test_images).hexdigest())
exp.set_property('test_labels_version', hashlib.md5(test_labels).hexdigest())

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

exp.set_property('class_names', class_names)

# Log images sample for each class
log_images_sample(class_names, train_labels, train_images, exp)

# Prepare model
model = get_model(model_params, optim_params)

# Log model summary
model.summary(print_fn=lambda x: exp.log_text('model_summary', x))

# Log model visualization
log_visualized_model(model, exp)

# Train model
model.fit(train_images, train_labels,
          batch_size=training_params.batch_size,
          epochs=training_params.n_epochs,
          shuffle=training_params.shuffle,
          callbacks=[
              tf.keras.callbacks.LambdaCallback(
                  on_epoch_end=lambda epoch, logs: log_epoch_data(logs, exp)),
              tf.keras.callbacks.LambdaCallback(
                  on_epoch_end=lambda epoch, logs:
                      log_model_weights(model, epoch, logs, exp) if epoch % training_params.save_every == 0 else False),
              tf.keras.callbacks.EarlyStopping(
                  patience=training_params.early_stopping,
                  monitor='accuracy',
                  restore_best_weights=True),
              tf.keras.callbacks.LearningRateScheduler(partial(lr_scheduler, npt=exp))]
          )

# Evaluate model
eval_metrics = model.evaluate(test_images, test_labels, verbose=0)
for j, metric in enumerate(eval_metrics):
    exp.log_metric('eval_' + model.metrics_names[j], metric)
