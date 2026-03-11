import matplotlib.pyplot as plt

def plot_history(history):

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])

    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend(["Train","Validation"])

    plt.show()