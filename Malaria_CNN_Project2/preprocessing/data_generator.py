import tensorflow as tf
from utils.config import BATCH_SIZE

def get_generators(train_data, val_data, train_labels_enc, val_labels_enc):

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.05,
        rotation_range=25,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(
        train_data,
        train_labels_enc,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_generator = val_datagen.flow(
        val_data,
        val_labels_enc,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_generator, val_generator