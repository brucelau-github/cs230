""" plot images """
import numpy as np
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
    x = np.linspace(0, 2, 100)
    images = [
        ["fiwdata/FIDs/F0008/MID1/P00102_face0.jpg",
         "fiwdata/FIDs/F0008/MID4/P00098_face2.jpg", "2"],
        ["fiwdata/FIDs/F0008/MID1/P00102_face0.jpg",
         "fiwdata/FIDs/F0008/MID4/P00098_face2.jpg", "2"]
    ]
    #plot_loss_accuracy(x, x**2)
    plot_images(images)

main()
