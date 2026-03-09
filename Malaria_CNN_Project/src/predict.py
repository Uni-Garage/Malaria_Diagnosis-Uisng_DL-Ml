import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224

model = load_model("../models/vgg_model.h5")

img_path = "sample_image.png"

img = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE))
img = image.img_to_array(img)/255
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

if prediction > 0.5:
    print("Parasitized Cell (Malaria Detected)")
else:
    print("Uninfected Cell")
