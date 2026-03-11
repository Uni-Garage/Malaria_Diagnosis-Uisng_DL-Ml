from sklearn.model_selection import train_test_split
from collections import Counter

def split_dataset(files_df):

    train_files, test_files, train_labels, test_labels = train_test_split(
        files_df["filename"].values,
        files_df["label"].values,
        test_size=0.3,
        random_state=42
    )

    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files,
        train_labels,
        test_size=0.1,
        random_state=42
    )

    print("Train:", Counter(train_labels))
    print("Validation:", Counter(val_labels))
    print("Test:", Counter(test_labels))

    return train_files, val_files, test_files, train_labels, val_labels, test_labels
