""" plot images """
#import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_images(images):
    """ plot images """
    dicts = {
        "0": "NAN",
        "1": "P-C",
        "2": "SIBS",
        "3": "SAME",
    }
    nrows = len(images)
    ncols = 2
    fig = plt.figure(figsize=(nrows, ncols))
    for i in range(nrows):
        fig.add_subplot(nrows, ncols, ncols * i + 1)
        plt.suptitle(dicts[images[i][2]])
        plt.axis('off')
        plt.imshow(cv2.imread(images[i][0]))
        fig.add_subplot(nrows, ncols, ncols * i + 2)
        plt.axis('off')
        plt.imshow(cv2.imread(images[i][1]))
    #plt.tight_layout()
    plt.show()

def plot_loss_accuracy(batches, loss):
    """ plot lost """
    _, axs = plt.subplots()  # Create a figure and an axes.
    axs.plot(batches, loss, label="loss")  # Plot some data on the axes.
    axs.set_xlabel("batches")  # Add an x-label to the axes.
    axs.set_ylabel("loss")  # Add a y-label to the axes.
    axs.set_title("Simple Plot")  # Add a title to the axes.
    axs.legend()  # Add a legend.
    plt.show()

def main():
    """ main """
    #x = np.linspace(0, 2, 100)
    images = [
        ["fiwdata/FIDs/F0008/MID1/P00102_face0.jpg",
         "fiwdata/FIDs/F0008/MID4/P00098_face2.jpg", "2"],
        ["fiwdata/FIDs/F0008/MID1/P00102_face0.jpg",
         "fiwdata/FIDs/F0008/MID4/P00098_face2.jpg", "2"]
    ]
    #plot_loss_accuracy(x, x**2)
    plot_images(images)

def plot_accuracy():
    """ plot accuracy """
    test_accuracy = [0.577500, 0.628333, 0.630667, 0.607000, 0.606500, 0.656000,
                     0.717833, 0.737833, 0.729167, 0.709667, 0.783333, 0.777500,
                     0.804000, 0.802167, 0.824333, 0.796333, 0.810667]
    train_accuracy = [0.459500, 0.539875, 0.628375, 0.610125, 0.698000,
                      0.755375, 0.635375, 0.712500, 0.759250, 0.652500, 0.753625,
                      0.807000, 0.693000, 0.793750, 0.842250, 0.705250, 0.819875,
                      0.867000, 0.730500, 0.801000, 0.850200, 0.769800, 0.862200,
                      0.777700, 0.878300, 0.783300, 0.872400, 0.795750, 0.870650,
                      0.815900, 0.886200, 0.830750, 0.894500, 0.844700, 0.908250,
                      0.850050, 0.909950, 0.857200, 0.920750, 0.872100, 0.922250]
    train_loss = [1.230838, 1.018025, 0.873610, 0.928813, 0.749189, 0.610484,
                  0.880050, 0.694358, 0.587156, 0.836249, 0.625494, 0.483024, 0.761195,
                  0.530135, 0.409667, 0.723385, 0.466743, 0.353218, 0.681323, 0.508410,
                  0.391199, 0.603397, 0.368984, 0.590649, 0.326144, 0.568664, 0.334087,
                  0.534612, 0.345774, 0.483523, 0.300826, 0.449272, 0.273585, 0.421462,
                  0.253014, 0.409101, 0.240019, 0.379063, 0.219109, 0.357845, 0.205774]
    _, axs = plt.subplots(1, 2)
    axs[0].set_title("test accuracy")
    axs[0].plot(test_accuracy, label="test accuracy")
    axs[1].set_title("train accuracy and loss")
    axs[1].plot(train_accuracy, label="train accuracy")
    axs[1].plot(train_loss, label="train loss")
    axs[1].set_xlabel("epochs")  # Add an x-label to the axes.
    axs[1].legend()  # Add a legend.
    plt.show()

plot_accuracy()
