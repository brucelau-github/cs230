""" preprocess image data """
import os
import sys
import pickle
import cv2
import pandas as pd

def load_parent_child(load_features=False):
    """ load parent child data """
    pickle_files = [
        #"fiwdata/lists/pairs/pickles/fd-faces.pkl",
        #"fiwdata/lists/pairs/pickles/fs-faces.pkl",
        #"fiwdata/lists/pairs/pickles/md-faces.pkl",
        "fiwdata/lists/pairs/pickles/ms-faces.pkl"
    ]

    data = pd.DataFrame([[], []], ["p1", "p2"])
    for file_path in pickle_files:
        print(file_path)
        with open(file_path, "rb") as pkl_file:
            data = data.append(pickle.load(pkl_file), ignore_index=True)

    fiw_base_dir = "FIDs"
    if load_features:
        fiw_base_dir = "FIDs-Features"
    fiw_base_dir = os.path.join("fiwdata", fiw_base_dir)
    if not os.path.exists(fiw_base_dir):
        print("{} doesn't exist".format(fiw_base_dir))
        sys.exit(1)

    pic = None
    for row in data.itertuples():
        _, pic1, pic2 = row
        pic = pic1
        pic = pic2

    img = cv2.imread(os.path.join(fiw_base_dir, pic), cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

load_parent_child()
