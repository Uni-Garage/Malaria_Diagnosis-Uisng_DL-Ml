import tensorflow as tf
from utils.config import INPUT_SHAPE

def build_vgg_finetuned():

    vgg = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE
    )

    vgg.trainable = True

    set_trainable = False

    for layer in vgg.layers:

        if layer.name in ["block5_conv1","block4_conv1"]:
            set_trainable = True

        layer.trainable = set_trainable

    x = tf.keras.layers.Flatten()(vgg.output)

    x = tf.keras.layers.Dense(512,activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(512,activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(1,activation="sigmoid")(x)

    model = tf.keras.Model(vgg.input,out)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model