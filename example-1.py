import neptune

neptune.init('neptune-ml/example-project')
neptune.create_experiment()

for i in range(100):
    neptune.log_metric('loss', 0.9**i)

neptune.stop()
