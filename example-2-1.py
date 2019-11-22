import neptune
import numpy as np

# select project
neptune.init('neptune-ml/example-project')

# create experiment
with neptune.create_experiment(name='get-started-example'):

    # log some metrics
    for i in range(1, 117):
        neptune.log_metric('iteration', i)
        neptune.log_metric('loss', 1/i**0.5)
        neptune.log_text('magic values', 'magic value {}'.format(0.95*i**2))

    # log property (key:value pairs)
    neptune.set_property('n_iterations', 117)

    # add tag to the experiment
    neptune.append_tag('example-2')

    # log some images
    for j in range(0, 5):
        array = np.random.rand(10, 10, 3)*255
        array = np.repeat(array, 30, 0)
        array = np.repeat(array, 30, 1)
        neptune.log_image('mosaics', array)
