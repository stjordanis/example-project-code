import neptune
import numpy as np

# select project
neptune.init('USERNAME/example-project')

# define parameters
PARAMS = {'timeseries_factor': 1.35,
          'n_iterations': 200,
          'n_images': 7}

# create experiment
with neptune.create_experiment(name='timeseries_example',
                               params=PARAMS):
    # log some metrics
    for i in range(1, PARAMS['n_iterations']):
        neptune.log_metric('iteration', i)
        neptune.log_metric('timeseries', PARAMS['timeseries_factor'] * np.cos(i / 10))
        neptune.log_metric('timeseries_sin', PARAMS['timeseries_factor'] * np.sin(i / 10))
        neptune.log_metric('timeseries_periodic_feature', PARAMS['timeseries_factor'] * np.exp(0.1 * ((i//30)*30 - i)))
        neptune.log_text('text_info', 'some value {}'.format(0.95 * i ** 2))

    # log property (key:value pair)
    neptune.set_property('timeseries_data_hash', '936f4232')

    # add tag to the experiment
    neptune.append_tag('timeseries_modeling')

    # log some images
    for j in range(PARAMS['n_images']):
        array = np.random.rand(10, 10, 3) * 255
        array = np.repeat(array, 30, 0)
        array = np.repeat(array, 30, 1)
        neptune.log_image('mosaics', array)
