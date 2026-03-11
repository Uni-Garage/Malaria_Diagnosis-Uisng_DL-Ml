from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image

# Load your trained model
model = load_model("models/vgg_finetuned.h5")

# Preprocess a sample parasitized cell image
img_array = preprocess_image("D:\Dinesh\Dinesh_test_files\PROJECTS\Me\FinalYEAR\Malaria_CNN_Project2\dataset\Malaria_dataset\Uninfected\C241NThinF_IMG_20151207_124643_cell_125.png")
# Make prediction
prob = model.predict(img_array)[0][0]

print("Predicted probability:", prob)

# Optional: map to label
prediction = "malaria" if prob > 0.5 else "healthy"
print("Prediction:", prediction)