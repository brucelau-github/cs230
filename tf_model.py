""""build a model to train kinship
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from load_dataset import load_image_pairs

def read_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize_with_pad(img, 299, 299)

def process_data(row):
    pic1, pic2, label = row[0], row[1], row[2]
    img1 = read_image(pic1)
    img2 = read_image(pic2)
    return img1, img2, label


def prepare_train():
    dataset = load_image_pairs()
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.map(process_data, num_parallel_calls=-1)
    dataset = dataset.cache(filename=".cache")
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(1000)
    dataset = dataset.prefetch(-1)

class KinNet(tf.keras.Model):
    def __init__(self):
        super(KinNet, self).__init__()
        self.inresnet = tf.keras.applications.InceptionResNetV2(
            include_top=False, weights='imagenet', input_shape=(299,299,3),
            pooling="avg")
        # output (1536, 1)
        self.dense1 = tf.keras.layers.Dense(4096, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(4096, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(4)

    def call(self, inputs, training=False):
        enc1 = self.inresnet(inputs[0])
        enc2 = self.inresnet(inputs[1])
        x = tf.layers.Concatenate(axis=1, name='concat')([enc1, enc2])
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def train():
    kinnet = KinNet()
    kinnet.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    kinnet.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)



