import glob
import pandas as pd
from utils.config import PARASITIZED_DIR, UNINFECTED_DIR

def load_dataset():

    infected_files = glob.glob(PARASITIZED_DIR + "/*.png")
    healthy_files = glob.glob(UNINFECTED_DIR + "/*.png")

    files_df = pd.DataFrame({
        "filename": infected_files + healthy_files,
        "label": ["malaria"] * len(infected_files) + ["healthy"] * len(healthy_files)
    })

    files_df = files_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return files_df