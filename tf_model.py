""""build a model to train kinship
"""
import os
import logging
import datetime
import random
import tensorflow as tf
import numpy as np
from load_dataset import load_image_pairs

def read_image(file_path):
    """ read image path in fiwdata
    resize it with pad
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize_with_pad(img, 299, 299)

def process_data(row):
    """ dataset call back
    read in image pair files
    covert into perper shape (1, 299, 299, 3)
    change labels to one hot encoding
    """
    pic1, pic2, label = row[0], row[1], row[2]
    img1 = read_image(pic1)
    img1 = tf.reshape(img1, [299, 299, 3])
    img2 = read_image(pic2)
    img2 = tf.reshape(img2, [299, 299, 3])
    label = tf.keras.backend.one_hot(int(label), 4)
    return [[img1, img2], label]

def logging_sample(data):
    """ print logging data """
    logging.info("data size: %d", len(data))
    np.random.shuffle(data)
    samples = random.sample(data, 10)
    logging.info(samples)

def prepare_train_data(batch_size):
    """ split train set and test set """
    train_set, test_set = load_image_pairs()
    logging.info("train set")
    logging_sample(train_set)
    logging.info("test set")
    logging_sample(test_set)

    dataset = tf.data.Dataset.from_tensor_slices(train_set)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(process_data, num_parallel_calls=-1)

    test_set = tf.data.Dataset.from_tensor_slices(test_set)
    test_set = test_set.shuffle(buffer_size=1000)
    valid_data = test_set.take(batch_size * 16)
    test_set = test_set.map(process_data, num_parallel_calls=-1)
    test_set = test_set.cache().shuffle(buffer_size=1000)
    test_set = test_set.batch(batch_size)
    test_set = test_set.prefetch(-1)

    valid_data = valid_data.map(process_data, num_parallel_calls=-1)
    valid_data = valid_data.cache().shuffle(buffer_size=1000)
    valid_data = valid_data.batch(batch_size)
    valid_data = valid_data.prefetch(-1)

    dataset = dataset.cache().shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(-1)

    return dataset, test_set, valid_data

class KinNet(tf.keras.Model):
    """ inception resnet + 2 dense """
    def __init__(self):
        super(KinNet, self).__init__()
        # output (batch_size, 1536)
        self.inresnet = tf.keras.applications.InceptionResNetV2(
            include_top=False, weights="imagenet", input_shape=(299, 299, 3),
            pooling="max")
        self.dense3 = tf.keras.layers.Dense(4, name="logits")

    def call(self, inputs, training=None, mask=None):
        enc1 = self.inresnet(inputs[:, 0, :, :, :])
        enc2 = self.inresnet(inputs[:, 1, :, :, :])
        x = tf.keras.layers.Concatenate(axis=1, name='concat')([enc1, enc2])
        x = self.dense3(x)
        return x

def initialize_logger(output_dir):
    """ init logger """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logformat = "%(asctime)s - %(levelname)s - %(message)s"

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(logformat)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(output_dir, "training.log"), "a")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(logformat)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class LossTracker(tf.keras.callbacks.Callback):
    """ log accuracy """
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            logging.info("batch %d: loss: %f acc: %f",
                         batch, logs.get('loss', -1.0), logs.get("accuracy", -1.0))

    def on_epoch_end(self, epoch, logs=None):
        logging.info("epoch %f: loss: %f, acc: %f",
                     epoch, logs.get("loss", -1.0), logs.get("accuracy", -1.0))


def train():
    """ train kinnet model """
    log_dir = "logs_{}".format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    initialize_logger(log_dir)

    train_set, test_set, validation_data = prepare_train_data(16)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir+"/cp-{epoch:04d}.ckpt", save_weights_only=False,
            save_freq='epoch'),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir+"/tensorboard"),
        LossTracker()
    ]

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    kinnet = KinNet()

    if os.path.exists("kinnet_weight.index"):
        kinnet.load_weights("kinnet_weight")
    kinnet.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    history = kinnet.fit(
        x=train_set, epochs=3, callbacks=callbacks, validation_data=validation_data)

    logging.info("history: %s", history.history)
    kinnet.save_weights("kinnet_weight")

    test_loss, test_acc = kinnet.evaluate(test_set, verbose=2, callbacks=[LossTracker()])
    logging.info("test accuracy: %f", test_acc)
    logging.info("test lost: %f", test_loss)

train()
