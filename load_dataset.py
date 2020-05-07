""" preprocess image data """
import os
import itertools
import math
import random
import pickle
import glob
import cv2
import numpy as np

def read_image(file_path):
    """ read image files """
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def read_pickle(pickle_file):
    """ read in pickle file """
    data = None
    with open(pickle_file, "rb") as pkl_file:
        data = pickle.load(pkl_file)
    return data

def parent_child_faces():
    """ list p-c pairs """
    pickle_files = [
        "fiwdata/lists/pairs/pickles/fd-faces.pkl",
        "fiwdata/lists/pairs/pickles/fs-faces.pkl",
        "fiwdata/lists/pairs/pickles/md-faces.pkl",
        "fiwdata/lists/pairs/pickles/ms-faces.pkl"
    ]


    base_dir = "fiwdata/FIDs-features/"
    face_pairs = []
    for pkl_file in pickle_files:
        data = read_pickle(pkl_file)
        for row in data.itertuples():
            _, file1, file2 = row
            pkl1 = base_dir + os.path.splitext(file1)[0] + ".pkl"
            pkl2 = base_dir + os.path.splitext(file2)[0] + ".pkl"
            face_pairs.append((pkl1, pkl2, 2))
    return face_pairs

def sibling_faces():
    """ list p-c pairs """
    pickle_files = [
        "fiwdata/lists/pairs/pickles/sibs-faces.pkl"
    ]


    base_dir = "fiwdata/FIDs-features/"
    face_pairs = []
    for pkl_file in pickle_files:
        data = read_pickle(pkl_file)
        for row in data.itertuples():
            _, file1, file2 = row
            pkl1 = base_dir + os.path.splitext(file1)[0] + ".pkl"
            pkl2 = base_dir + os.path.splitext(file2)[0] + ".pkl"
            face_pairs.append((pkl1, pkl2, 2))
    return face_pairs

def sames_diff_faces():
    """list all same faces data set
    return: [(f1, f2), (f1, f3) ....]
    """
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

    face_pairs = []
    for same_face in same_faces:
        permu = itertools.permutations(same_face, 2)
        for pair in permu:
            face_pairs.append((pair[0], pair[1], 3))

    for diff_face in diff_faces:
        produ = itertools.product(diff_face[0], diff_face[1])
        for pair in produ:
            face_pairs.append((pair[0], pair[1], 0))

    return face_pairs

def print_sample(data):
    samples = random.sample(data, 50)
    for i in samples:
        print(i)
    print("contains: %d data"%len(data))

def load_faces():
    """ load features """
    file_name = "face_pairs.pkl"
    face_pairs = None
    if os.path.isfile(file_name):
        face_pairs = read_pickle(file_name)
    if not face_pairs:
        face_pairs = parent_child_faces()
        face_pairs.extend(sibling_faces())
        face_pairs.extend(sames_diff_faces())
        random.shuffle(face_pairs)
        save_pickle(face_pairs, file_name)
    return face_pairs

def save_pickle(data, file_name="face_pairs.pkl"):
    """ save data as pickle """
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=2)

def convert_numpy():
    """ convert face pairs to numpy array """
    file_name = "data_matrix.npy"
    label_name = "label_matrix.npy"
    data_matrix, label_matrix = None, None
    if os.path.isfile(file_name):
        data_matrix, label_matrix = np.load(file_name), np.load(label_name)
    label_dict = {
        0: [1, 0, 0, 0],
        1: [0, 1, 0, 0],
        2: [0, 0, 1, 0],
        3: [0, 0, 0, 1]
    }
    if data_matrix is None:
        face_pairs = load_faces()
        labels = []
        data = []
        for pair in face_pairs:
            file1, file2, label = pair
            enc1 = read_pickle(file1)
            enc2 = read_pickle(file2)
            data.append(np.concatenate((enc1, enc2)))
            labels.append(label_dict[label])
        data_matrix = np.array(data).T
        label_matrix = np.array(labels).T
        np.save(file_name, data_matrix)
        np.save(label_name, label_matrix)

    return (data_matrix, label_matrix)

def split_test_data(data, labels, ratio=0.05):
    """ split and shuffle data """
    para_m = data.shape[1]
    shuffle = np.arange(para_m)
    np.random.shuffle(shuffle)
    shuffled_x = data[:, shuffle]
    shuffled_y = labels[:, shuffle]
    pivot = math.floor(para_m * (1 - ratio))
    return (shuffled_x[:, :pivot], shuffled_y[:, :pivot],
            shuffled_x[:, pivot:], shuffled_y[:, pivot:])

def load_dataset():
    """ return train_x, train_y, test_x, test_y """
    data, labels = convert_numpy()
    file_list = [
        ["train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"],
        []
    ]
    for idx in range(4):
        if os.path.isfile(file_list[0][idx]):
            file_list[1].append(np.load(file_list[0][idx]))

    if len(file_list[1]) != 4:
        data = split_test_data(data, labels)
        for idx in range(4):
            file_list[1].append(data[idx])
            np.save(file_list[0][idx], data[idx])

    return tuple(file_list[1])

def load_small_testdata():
    """ load a small data """
    data, labels = convert_numpy()
    data = data[:, :20000]
    labels = labels[:, :20000]
    file_list = [
        ["train_x_s.npy", "train_y_s.npy", "test_x_s.npy", "test_y_s.npy"],
        []
    ]
    for idx in range(4):
        if os.path.isfile(file_list[0][idx]):
            file_list[1].append(np.load(file_list[0][idx]))

    if len(file_list[1]) != 4:
        data = split_test_data(data, labels)
        for idx in range(4):
            file_list[1].append(data[idx])
            np.save(file_list[0][idx], data[idx])

    return tuple(file_list[1])
