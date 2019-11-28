import neptune

neptune.init('kamil/example-project')
neptune.create_experiment(name='bare_minimal_example')

for i in range(100):
    neptune.log_metric('loss', 0.9**i)

neptune.stop()
