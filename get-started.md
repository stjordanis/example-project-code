# Get started
**Installation**

`pip install neptune-client`

**Run script**

Copy snippet below to "get-started.py" and run this file
```
import neptune
import numpy as np

# select project
neptune.init('USERNAME/example-project',
             api_token='<NEPTUNE_API_TOKEN>')

# create experiment
neptune.create_experiment(name='quick-start-example')

# log some metrics
for i in range(1, 117):
    neptune.log_metric('iteration', i)
    neptune.log_metric('loss', 1/i**0.5)
    neptune.log_text('magic values', 'magic value {}'.format(0.95*i**2))

# log property (key:value pairs)
neptune.set_property('n_iterations', 117)

# log some images
for j in range(0, 5):
    array = np.random.rand(10, 10, 3)*255
    array = np.repeat(array, 30, 0)
    array = np.repeat(array, 30, 1)
    neptune.log_image('mosaics', array)

neptune.stop()
```
Go to https://ui.neptune.ml/<USERNAME>/example-project/experiments to check results!

Your experiment is at the top of the table. This experiment becomes a part of the **example project**.
Go to the Wiki section that will help you get started: https://ui.neptune.ml/<USERNAME>/example-project/wiki
