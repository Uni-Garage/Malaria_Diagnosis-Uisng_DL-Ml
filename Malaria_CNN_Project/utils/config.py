import os

BASE_DIR = "dataset/Malaria_dataset"

PARASITIZED_DIR = os.path.join(BASE_DIR, "Parasitized")
UNINFECTED_DIR = os.path.join(BASE_DIR, "Uninfected")

INPUT_SHAPE = (125,125,3)

BATCH_SIZE = 64
EPOCHS = 25
NUM_CLASSES = 2

MODEL_SAVE_PATH = "saved_models/"
