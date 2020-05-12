""""build a model to train kinship
"""
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
    img1 = tf.reshape(img1, [1, 299, 299, 3])
    img2 = read_image(pic2)
    img2 = tf.reshape(img2, [1, 299, 299, 3])
    label = tf.keras.backend.one_hot(int(label), 4)
    label = tf.reshape(label, [1, 4])
    return [[img1, img2], label]

def prepare_train_data():
    """ split train set and test set """
    dataset = load_image_pairs()
    test_size = int(0.05*len(dataset))
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.map(process_data, num_parallel_calls=-1)
    test_set = dataset.take(test_size)
    test_set.cache()
    dataset = dataset.cache(filename=".cache").shuffle(buffer_size=1000)
    dataset = dataset.prefetch(-1)

    return dataset, test_set

class KinNet(tf.keras.Model):
    """ inception resnet + 2 dense """
    def __init__(self):
        super(KinNet, self).__init__()
        self.inresnet = tf.keras.applications.InceptionResNetV2(
            include_top=False, weights='imagenet', input_shape=(299, 299, 3),
            pooling="avg")
        # output (1536, 1)
        self.dense1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(4)

    def call(self, inputs, training=None, mask=None):
        enc1 = self.inresnet(inputs[0, :, :, :, :])
        enc2 = self.inresnet(inputs[1, :, :, :, :])
        x = tf.keras.layers.Concatenate(axis=1, name='concat')([enc1, enc2])
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def train():
    """ train kinnet model """
    train_set, test_set = prepare_train_data()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoint", save_weights_only=True,
            monitor="val_acc", mode="max", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir="./logs")
    ]


    kinnet = KinNet()

    kinnet.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    kinnet.fit(x=train_set, epochs=10, callbacks=callbacks)
    kinnet.save_weights("kinnet_weight.h5", save_format="h5")
    #kinnet.load_weights("my_model.h5")

    test_loss, test_acc = kinnet.evaluate(test_set, verbose=2)
    print("\nTest accuracy:", test_acc)
    print("\nTest lost:", test_loss)

train()
