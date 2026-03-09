from data_preprocessing import load_data
from vgg_model import build_vgg
from resnet_model import build_resnet
import matplotlib.pyplot as plt

train_data, test_data = load_data()

# Train VGG16
print("Training VGG16 model...")
vgg = build_vgg()

history_vgg = vgg.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

vgg.save("../models/vgg_model.h5")


# Train ResNet50
print("Training ResNet50 model...")
resnet = build_resnet()

history_resnet = resnet.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

resnet.save("../models/resnet_model.h5")


# Accuracy Plot
plt.plot(history_vgg.history["accuracy"], label="VGG Train")
plt.plot(history_vgg.history["val_accuracy"], label="VGG Val")

plt.plot(history_resnet.history["accuracy"], label="ResNet Train")
plt.plot(history_resnet.history["val_accuracy"], label="ResNet Val")

plt.legend()
plt.title("Model Accuracy Comparison")
plt.savefig("../results/accuracy_plot.png")
plt.show()
