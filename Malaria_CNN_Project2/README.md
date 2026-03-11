🦠 Malaria Cell Detection using CNN and Flask Web Application

📌 Overview
Malaria is a life-threatening disease caused by parasites transmitted through mosquito bites. Early and accurate detection of malaria-infected blood cells is crucial for effective treatment.

This project builds an end-to-end deep learning pipeline for malaria detection using:

Convolutional Neural Networks (CNN)

Transfer Learning with VGG19

TensorFlow / Keras

Flask Web Application

The system allows users to upload microscope cell images and receive real-time predictions indicating whether the cell is parasitized (malaria) or healthy.

🚀 Features

✔ CNN-based malaria detection
✔ Transfer learning with VGG19
✔ Image preprocessing pipeline
✔ Model evaluation with Accuracy, Precision, Recall, F1-score
✔ Flask-based web interface
✔ Image upload and prediction
✔ Confidence visualization
✔ Extensible architecture for Grad-CAM explainability

📂 Project Structure
malaria_detection/
│
├── dataset/
│   └── Malaria_dataset/
│       ├── Parasitized/
│       └── Uninfected/
│
├── models/
│   ├── cnn_scratch.py
│   ├── vgg_finetuned.py
│   └── vgg_frozen.py
│
├── preprocessing/
│   ├── data_generator.py
│   ├── load_dataset.py
│   ├── split_dataset.py
│   └── load_images.py
│
├── utils/
│   ├── config.py
│   └── plot_metrics.py
├── train.py
├── evaluate.py
├── app.py
└── README.md

malaria_web_app_2/
│
├── templates/
│   └── index.html
├── models/
│   └── vgg_finetuned.h5
├── static/
│   └── uploads/
├── utils/
│   ├── config.py
│   └── preprocess.py
├── test_model.py
├── app.py
└── requirements.txt

🧠 Dataset

Dataset used:
Cell Images for Detecting Malaria

Source:
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

Classes
Class	Description
Parasitized	Malaria infected cells
Uninfected	Healthy blood cells
Input Image Size
125 x 125 x 3 (RGB)
🔧 Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/malaria-detection-cnn.git
cd malaria-detection-cnn
2️⃣ Create Virtual Environment
python -m venv env
source env/bin/activate   # Linux/Mac
env\Scripts\activate      # Windows
3️⃣ Install Dependencies
pip install tensorflow flask numpy pandas scikit-learn matplotlib
🧪 Model Architectures

The project includes three models:

1️⃣ Custom CNN

A simple CNN built from scratch.

Architecture:

Input (125x125x3)
↓
Conv2D (32)
↓
MaxPooling
↓
Conv2D (64)
↓
MaxPooling
↓
Conv2D (128)
↓
MaxPooling
↓
Flatten
↓
Dense (512)
↓
Dropout
↓
Dense (512)
↓
Dropout
↓
Sigmoid Output
2️⃣ VGG19 Frozen

Uses VGG19 pretrained on ImageNet as a feature extractor.

Convolution layers frozen

Custom classifier layers added

3️⃣ VGG19 Fine-Tuned

Improved version where last VGG layers are unfrozen.

Benefits:

Better feature adaptation

Higher prediction accuracy

⚙️ Training the Model

Run the training script:

python train.py

Training process:

1️⃣ Load dataset
2️⃣ Split into train / validation / test
3️⃣ Convert images to arrays
4️⃣ Normalize pixel values
5️⃣ Train CNN or VGG19 model
6️⃣ Save trained model

Saved model:

saved_models/vgg_finetuned.h5
📊 Model Evaluation

Run evaluation:

python evaluate.py

Metrics computed:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Example output:

Accuracy: 0.96

Classification Report

              precision    recall  f1-score

healthy          0.97       0.96       0.96
malaria          0.95       0.96       0.95
🌐 Web Application

The project includes a Flask web app for real-time predictions.

Users can:

✔ Upload microscope image
✔ Get prediction instantly
✔ View confidence score
✔ See uploaded image preview

🖥 Running the Web App

Start the Flask server:

python app.py

Open browser:

http://127.0.0.1:5000

Upload a cell image and view prediction.

🔄 Internal Workflow

The prediction pipeline follows these steps:

User Upload Image
        │
        ▼
Flask receives file
        │
        ▼
Image preprocessing
(resize + normalize)
        │
        ▼
CNN / VGG19 Model
        │
        ▼
Probability Output (Sigmoid)
        │
        ▼
Threshold Classification
        │
        ▼
Prediction + Confidence
        │
        ▼
Displayed on Web Interface
📈 Prediction Output

Example output:

Prediction: malaria
Confidence: 87%

Or

Prediction: healthy
Confidence: 92%

If probability is near threshold:

Prediction: malaria (uncertain)
🔬 Future Improvements

Potential extensions:

Grad-CAM visualization

Multi-image batch prediction

Model comparison dashboard

Cloud deployment

Mobile interface

Real-time microscope integration

📚 Technologies Used
Technology	Purpose
TensorFlow / Keras	Deep Learning
CNN	Feature extraction
VGG19	Transfer Learning
Flask	Web Application
NumPy / Pandas	Data processing
Scikit-learn	Model evaluation
Matplotlib	Visualization
📷 Example Prediction Interface
Upload Image
      ↓
+----------------------+
|      Cell Image      |
+----------------------+

Prediction: Malaria
Confidence: 91%

[Confidence Bar]
👨‍💻 Author

Developed as a Deep Learning + Medical Image Classification Project demonstrating:

Computer Vision

Transfer Learning

Model Deployment

Web Application Integration

📜 License

This project is intended for educational and research purposes.
