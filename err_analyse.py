""""
load model and get all false negative priduction
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import InceptionResNetV2

from load_dataset import load_image_pairs
from tf_model import read_image

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


def predict():
    """ predict """
    _, test_pairs = load_image_pairs()
    pic1_list = []
    pic2_list = []
    label_list = []
    for row in test_pairs[:10]:
        pic1, pic2, label = row[0], row[1], row[2]
        img1 = read_image(pic1)
        img1 = tf.reshape(img1, [299, 299, 3])
        pic1_list.append(img1)
        img2 = read_image(pic2)
        img2 = tf.reshape(img2, [299, 299, 3])
        pic2_list.append(img2)
        label = tf.keras.backend.one_hot(int(label), 4)
        label_list.append(label)
    model = load_model("kinnet_model.hd5")
    print(model([pic1_list, pic2_list]))

predict()
