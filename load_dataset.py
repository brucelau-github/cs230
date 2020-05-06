""" preprocess image data """
import os
import itertools
import random
import sys
import pickle
import glob
import cv2
import pandas as pd

def load_dataset(load_features=False):
    """ load siblings data """
    datasets = [
        {
            "pickle_files": [
                #"fiwdata/lists/pairs/pickles/fd-faces.pkl",
                #"fiwdata/lists/pairs/pickles/fs-faces.pkl",
                #"fiwdata/lists/pairs/pickles/md-faces.pkl",
                "fiwdata/lists/pairs/pickles/ms-faces.pkl"
            ],
            "label": 1
        },
        {
            "pickle_files": ["fiwdata/lists/pairs/pickles/sibs-faces.pkl"],
            "label": 2
        }
    ]
    data = []
    for value in datasets:
        print(value)
        data.extend(load_data(load_features, value["pickle_files"], value["label"]))
    print(len(data))
    print(data[0])

def load_data(load_features, pickle_files, label):
    """ load parent child data """
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
            raw_data.append((read_feature(pkl1), read_feature(pkl2), label))
        else:
            raw_data.append((read_image(file1), read_image(file2), label))

    return raw_data

def read_image(file_path):
    """ read image files """
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def read_feature(pickle_file):
    """ read in pickle file """
    data = None
    with open(pickle_file, "rb") as pkl_file:
        data = pickle.load(pkl_file)
    return data

def sames_faces():
    """list all same faces data set
    return: [(f1, f2), (f1, f3) ....]
    """
    # walk folder path
    base_dir = "fiwdata/FIDs-features"
    same_faces = []
    diff_faces = []
    for fid in glob.glob(base_dir+"/F*"):
        tmp = []
        for mid in glob.glob(fid+"/MID*"):
            mid_faces = glob.glob(mid+"/*.pkl")
            same_faces.append(mid_faces)
            tmp.extend(mid_faces)
        diff_faces.append([tmp, glob.glob(fid+"/unrelated_and_nonfaces/*.pkl")])
        if len(diff_faces) > 2:
            break
            # mid_dir = os.path.join(fid_dir, mid)
            # if "unrelated" in mid:
                # print(glob.glob(mid_dir+"/*.pkl"))
            #     family_mid.append(os.listdir(mid))
        #     else:
        #         same_faces.append(os.listdir(mid))
    same_face_pairs = []
    for same_face in same_faces:
        permu = itertools.permutations(same_face, 2)
        for pair in permu:
            same_face_pairs.append(pair)

    diff_face_pairs = []
    for diff_face in diff_faces:
        produ = itertools.product(diff_face[0], diff_face[1])
        for pair in produ:
            diff_face_pairs.append(pair)

    print_sample(random.sample(same_face_pairs, 5))
    print_sample(random.sample(diff_face_pairs, 5))

def print_sample(samples):
    for i in samples:
        print(i[0])
        print(i[1])
        print()
sames_faces()
