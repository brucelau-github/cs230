""""
load model and get all false negative priduction
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import InceptionResNetV2

from load_dataset import load_image_pairs
from tf_model import process_data, KinNet, read_image

def kin_net(input_shape, weights=None):
    """Kin net model"""
    img_input1 = Input(shape=input_shape)
    img_input2 = Input(shape=input_shape)
    encoder = InceptionResNetV2(
        include_top=False, input_shape=input_shape,
        pooling="max")
    encoder.set_weights(weights["kinnet_inception_weights"])
    x = Concatenate(axis=1, name='concat')(
        [encoder(img_input1), encoder(img_input2)])
    outputs = Dense(4, weights=weights["kinnet_logits_weights"], name="logits")(x)
    model = Model(inputs=[img_input1, img_input2], outputs=outputs)
    return model


def construct_model():
    """ predict """
    _, test_pairs = load_image_pairs()
    test_set = tf.data.Dataset.from_tensor_slices(test_pairs[:10])
    test_set = test_set.map(process_data, num_parallel_calls=-1)
    test_set = test_set.batch(2)

    ini_set = tf.data.Dataset.from_tensor_slices(test_pairs[:1])
    ini_set = ini_set.map(process_data, num_parallel_calls=-1).batch(1)
    # parent-child pairs
    x = read_images('fiwdata/FIDs/F0729/MID2/P11068_face2.jpg',
                    'fiwdata/FIDs/F0729/MID5/P11064_face0.jpg')
    opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
    kinnet = KinNet()
    print("loading model from checkpoint")
    kinnet.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    kinnet.fit(x=ini_set, epochs=1)
    kinnet.load_weights(tf.train.latest_checkpoint("logs_20200525_190033"))
    weights = {
        "kinnet_inception_weights": kinnet.inresnet.get_weights(),
        "kinnet_logits_weights": kinnet.dense3.get_weights()
    }
    model = kin_net((299, 299, 3), weights)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    model.save("kinnet_model.hd5")
    print(model(x))

def predict():
    """ saved model """
    model = load_model("kinnet_model.hd5")
    x = read_images('fiwdata/FIDs/F0729/MID2/P11068_face2.jpg',
                    'fiwdata/FIDs/F0729/MID5/P11064_face0.jpg')
    print(model(x))

def read_images(pic1, pic2):
    """ read image """
    img1 = read_image(pic1)
    img1 = tf.reshape(img1, [1, 299, 299, 3])
    img2 = read_image(pic2)
    img2 = tf.reshape(img2, [1, 299, 299, 3])
    return [[img1, img2]]
predict()
