import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing.load_dataset import load_dataset
from preprocessing.split_dataset import split_dataset
from preprocessing.load_images import load_images

# Load dataset
files_df = load_dataset()
train_files, val_files, test_files, train_labels, val_labels, test_labels = split_dataset(files_df)

# Encode labels
le = LabelEncoder()
le.fit(train_labels)

test_labels_enc = le.transform(test_labels)

# Load test images
print("Loading test images...")
test_data = load_images(test_files)

# Load trained model
print("Loading model...")
model = load_model("./models/vgg_finetuned.h5")

# Evaluate model
print("Evaluating model...")
loss, accuracy = model.evaluate(test_data, test_labels_enc)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)

# Predictions
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels_enc, y_pred, target_names=le.classes_))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(test_labels_enc, y_pred))