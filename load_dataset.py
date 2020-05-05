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

    data = pd.DataFrame()
    for file_path in pickle_files:
        with open(file_path, "rb") as pkl_file:
            data = data.append(pickle.load(pkl_file), ignore_index=True)

    fiw_base_dir = "FIDs"
    if load_features:
        fiw_base_dir = "FIDs-features"
    fiw_base_dir = os.path.join("fiwdata", fiw_base_dir)
    if not os.path.exists(fiw_base_dir):
        print("{} doesn't exist".format(fiw_base_dir))
        sys.exit(1)

    raw_data = []
    for row in data.itertuples():
        _, file1, file2 = row
        file1 = os.path.join(fiw_base_dir, file1)
        file2 = os.path.join(fiw_base_dir, file2)
        if load_features:
            pkl1 = os.path.splitext(file1)[0] + ".pkl"
            pkl2 = os.path.splitext(file2)[0] + ".pkl"
            raw_data.append((read_feature(pkl1), read_feature(pkl2)))
        else:
            raw_data.append((read_image(file1), read_image(file2)))

    print(len(raw_data))

def read_image(file_path):
    """ read image files """
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def read_feature(pickle_file):
    """ read in pickle file """
    data = None
    with open(pickle_file, "rb") as pkl_file:
        data = pickle.load(pkl_file)
    return data

load_parent_child(True)
load_parent_child()
