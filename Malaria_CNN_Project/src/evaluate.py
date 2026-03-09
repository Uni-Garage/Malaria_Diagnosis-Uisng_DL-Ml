from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_data

train_data, test_data = load_data()

model = load_model("../models/vgg_model.h5")

pred = model.predict(test_data)
pred = np.round(pred)

cm = confusion_matrix(test_data.classes, pred)

print("Confusion Matrix")
print(cm)

print("Classification Report")
print(classification_report(test_data.classes, pred))


plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

classes = ["Parasitized","Uninfected"]

plt.xticks(range(2), classes)
plt.yticks(range(2), classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("../results/confusion_matrix.png")
plt.show()
