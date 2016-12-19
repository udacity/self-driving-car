This is a snapshot of the repository github.com/emef/sdc

-----

This repo contains the code used to generate models and submissions
for udacity's self driving car challenge 2. Our team name in the
leaderboards was 'chauffeur'.

-----

## Disclaimer

A good portion of this code is commented but not all of it was used in
the later stages of the competition. This is a high-level breakdown of
our interfaces.

## Dependencies

tensorflow, keras, opencv, numpy, scipy, requests, progress...

and probably some others I'm forgetting as well.

### Datasets

A dataset directory should have this structure:

```
  ./training_indexes.npy
  ./testing_indexes.npy
  ./validation_indexes.npy
  ./labels.npy
  ./images/
      1.png.npy
      2.png.npy
      ...
      N.png.npy
```

**Note that the image numpy files are 1-indexed.**

### Models

We added some abstraction on top of keras models to simplify training
and evaluating sdc datasets, saving and loading, nesting models for
transfer learning, etc.

### Training

```python
from callbacks import SnapshotCallback
from datasets import load_dataset
from models import load_from_config, RegressionModel

# RegressionModel is a cnn that directly predicts the steering angle
init_model_config = RegressionModel.create(
  '/tmp/regression_model.keras',
  use_adadelta=True,
  learning_rate=0.001,
  input_shape=(120, 320, 3))
  
model = load_from_config(init_model_config)

# this path contains a dataset in the prescribed format
dataset = load_dataset('/datasets/showdown_full')

# snapshots the model after each epoch
snapshot = SnapshotCallback(
  model,
  snapshot_dir='/tmp/snapshots/',
  score_metric='val_rmse')

model.fit(dataset, {
    'batch_size': 32,
    'epochs': 40,
  },
  final=False,  # don't train on the test holdout set
  callbacks=[snapshot])

# save model to local file and return the 'config' so it can be loaded
model_config = model.save('/tmp/regression.keras')

# evaluate the model on the test holdout
print model.evaluate(dataset)
```

### Generating a video on the test set with overlayed steering anle

This is pretty easy, it's all automated in the program
submission.py. You will need mencoder and ffmpeg installed on your
machine.