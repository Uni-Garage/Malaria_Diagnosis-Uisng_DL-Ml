import tensorflow as tf
from utils.config import INPUT_SHAPE

def build_cnn():

    inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

    conv1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(inp)
    pool1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)

    conv2 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)

    conv3 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D((2,2))(conv3)

    flat = tf.keras.layers.Flatten()(pool3)

    dense1 = tf.keras.layers.Dense(512,activation='relu')(flat)
    drop1 = tf.keras.layers.Dropout(0.3)(dense1)

    dense2 = tf.keras.layers.Dense(512,activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(0.3)(dense2)

    out = tf.keras.layers.Dense(1,activation='sigmoid')(drop2)

    model = tf.keras.Model(inp,out)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model