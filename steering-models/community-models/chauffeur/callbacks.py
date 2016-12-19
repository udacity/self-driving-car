import logging, os, time

from keras.callbacks import Callback

logger = logging.getLogger(__name__)


class SnapshotCallback(Callback):
    """
    Callback which saves the model snapshot to s3 on each epoch
    """
    def __init__(self,
                 model_to_save,
                 snapshot_dir,
                 score_metric='val_rmse'):
        self.model_to_save = model_to_save
        self.snapshot_dir = snapshot_dir
        self.score_metric = score_metric
        self.min_score = None
        self.min_path = None
        self.max_score = None
        self.max_path = None
        self.nb = 0

        logger.info('Saving snapshots to %s', snapshot_dir)

    def on_epoch_end(self, epoch, logs):
        model_path = os.path.join(self.snapshot_dir, '%d.h5' % self.nb)
        score = logs.get(self.score_metric)
        if self.min_score is None or score < self.min_score:
            self.min_score = score
            self.min_path = model_path

        if self.max_score is None or score > self.max_score:
            self.max_score = score
            self.max_path = model_path

        self.model_to_save.save(model_path)
        self.nb += 1

        logger.info('Snapshotted model to %s', model_path)


class TimedEarlyStopping(Callback):
    """
    Stop training after N minutes.
    """
    def __init__(self, duration_minutes):
        self.duration = duration_minutes
        self.started_at_ts = time.time()

    def on_batch_end(self, batch, logs):
        delta_ts = time.time() - self.started_at_ts
        delta_minutes = delta_ts / 60
        if delta_minutes >= self.duration:
            self.model.stop_training = True
