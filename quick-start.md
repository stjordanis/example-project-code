# Quick start
**Installation**

`pip install neptune-client`

**Run script**

Copy snippet below to "quick-start.py" and run this file
```
import neptune
import numpy as np

# select project
neptune.init('<USERNAME>/example-project',
             api_token='<NEPTUNE_API_TOKEN>')

# define parameters
PARAMS = {'magic_factor': 0.5,
          'n_iterations': 117}

# create experiment
neptune.create_experiment(name='quick_start_example',
                          params=PARAMS)

# log some metrics
for i in range(1, PARAMS['n_iterations']):
    neptune.log_metric('iteration', i)
    neptune.log_metric('loss', PARAMS['magic_factor']/i**0.5)
    neptune.log_text('magic_values', 'magic value {}'.format(0.95*i**2))

# add tag to the experiment
neptune.append_tag('quick_start')

# log some images
for j in range(5):
    array = np.random.rand(10, 10, 3)*255
    array = np.repeat(array, 30, 0)
    array = np.repeat(array, 30, 1)
    neptune.log_image('mosaics', array)

neptune.stop()
```
Go to https://ui.neptune.ml/<USERNAME>/example-project/experiments to check results!

Your experiment is at the top of the table. This experiment becomes a part of the **example project**.
Go to the Wiki section that will help you get started: https://ui.neptune.ml/<USERNAME>/example-project/wiki
