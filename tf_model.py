""""build a model to train kinship
"""
import os
import logging
import tensorflow as tf
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

def prepare_train_data(batch_size):
    """ split train set and test set """
    dataset = load_image_pairs()
    data_size = len(dataset)
    test_size = int(0.05*data_size)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(process_data, num_parallel_calls=-1)

    test_set = dataset.take(test_size)
    test_set = test_set.cache().shuffle(buffer_size=1000)
    test_set = test_set.batch(batch_size)
    test_set = test_set.prefetch(-1)

    valid_data = dataset.take(128)
    valid_data = valid_data.cache().shuffle(buffer_size=1000)
    valid_data = valid_data.batch(batch_size)
    valid_data = valid_data.prefetch(-1)

    dataset = dataset.cache().shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(-1)
    logging.info("contain %d data: %d train, %d test", data_size,
                 data_size-test_size, test_size)

    return dataset, test_set, valid_data

class KinNet(tf.keras.Model):
    """ inception resnet + 2 dense """
    def __init__(self):
        super(KinNet, self).__init__()
        self.inresnet = tf.keras.applications.InceptionResNetV2(
            include_top=False, weights='imagenet', input_shape=(299, 299, 3),
            pooling="avg")
        # output (1536, 1)
        self.dense1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu, name="fc1")
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu, name="fc2")
        self.dense3 = tf.keras.layers.Dense(4, name="logits")

    def call(self, inputs, training=None, mask=None):
        enc1 = self.inresnet(inputs[:, 0, :, :, :])
        enc2 = self.inresnet(inputs[:, 1, :, :, :])
        x = tf.keras.layers.Concatenate(axis=1, name='concat')([enc1, enc2])
        x = self.dense1(x)
        x = self.dense2(x)
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

def train():
    """ train kinnet model """
    checkpoint_path = "results/cp-{epoch:04d}.ckpt"
    result_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    initialize_logger(result_dir)

    train_set, test_set, validation_data = prepare_train_data(16)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True,
            save_freq='epoch'),
        tf.keras.callbacks.TensorBoard(log_dir="./logs")
    ]


    kinnet = KinNet()

    #kinnet.load_weights("kinnet_weight.h5")
    kinnet.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    history = kinnet.fit(
        x=train_set, epochs=1, callbacks=callbacks, validation_data=validation_data)

    logging.info("history: %s", history.history)
    logging.info("saving weights files: kinnet_weight.h5")
    kinnet.save_weights("kinnet_weight.h5", save_format="h5")

    test_loss, test_acc = kinnet.evaluate(test_set, verbose=2)
    logging.info("test accuracy: %f", test_acc)
    logging.info("test lost: %f", test_loss)

train()
