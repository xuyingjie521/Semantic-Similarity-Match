import math
import tensorflow as tf


def create_learning_rate_scheduler(config, verbose=0):
    learning_rate = config.learning_rate
    warmup_epochs = config.warmup_epochs
    lr_decay_rate = config.lr_decay_rate
    lr_decay_epoch = config.lr_decay_epoch
    min_learning_rate = config.min_learning_rate

    if verbose == 1:
        print("learning_rate:", learning_rate)
        print("warmup_epochs:", warmup_epochs)
        print("lr_decay_rate:", lr_decay_rate)
        print("lr_decay_epoch:", lr_decay_epoch)
        print("min_learning_rate:", min_learning_rate)

    def lr_scheduler(epoch):
        if epoch < warmup_epochs:
            lr = (learning_rate / warmup_epochs) * (epoch + 1)
        else:
            lr = max(min_learning_rate, learning_rate * math.pow(lr_decay_rate, math.floor((1 + epoch - warmup_epochs) / lr_decay_epoch)))
        return float(lr)

    scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    return scheduler
