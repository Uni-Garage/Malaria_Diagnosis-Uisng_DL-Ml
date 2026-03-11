from preprocessing.load_dataset import load_dataset
from preprocessing.split_dataset import split_dataset
from preprocessing.load_images import load_images
from models.vgg_finetuned import build_vgg_finetuned
from sklearn.preprocessing import LabelEncoder

# Load dataset
files_df = load_dataset()
train_files, val_files, test_files, train_labels, val_labels, test_labels = split_dataset(files_df)

# Convert labels to numbers
le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
val_labels_enc = le.transform(val_labels)

# Load images into numpy arrays
train_data = load_images(train_files)
val_data = load_images(val_files)

# Build model
model = build_vgg_finetuned()
model.summary()

# Train model
history = model.fit(
    train_data,
    train_labels_enc,
    batch_size=64,
    epochs=25,
    validation_data=(val_data, val_labels_enc)
)

# Save model
model.save("models/vgg_finetuned.h5")
