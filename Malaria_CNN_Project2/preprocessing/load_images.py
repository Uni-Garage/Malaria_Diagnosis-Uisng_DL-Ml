import numpy as np
from tensorflow.keras.preprocessing import image
from utils.config import INPUT_SHAPE

def load_images(file_paths):
    imgs = []
    for fp in file_paths:
        img = image.load_img(fp, target_size=INPUT_SHAPE[:2])
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0   # normalize
        imgs.append(img_array)
    return np.array(imgs)