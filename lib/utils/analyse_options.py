import os
import scipy
import nrrd
import glob
import nibabel as nib
import seaborn as sns




def show_space_property(dataset_path=r"../../datasets/src_10"):
    train_images = glob.glob(os.path.join(dataset_path, 'train', "images", '*.nrrd'))
    train_labels = glob.glob(os.path.join(dataset_path, 'train', "labels", '*.nrrd'))
    val_images = glob.glob(os.path.join(dataset_path, 'val', "images", '*.nrrd'))
    val_labels = glob.glob(os.path.join(dataset_path, 'val', "labels", '*.nrrd'))
    keys = ["type", "dimension", "space", "sizes", "space directions", "space origin"]
    values = []
    for i, key in enumerate(keys):
        values.append([])
        for path in train_images:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        for path in train_labels:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        for path in val_images:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        for path in val_labels:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        print(key + ": ", values[i])



def analyse_dataset(dataset_path=r"../../datasets/src_10"):
    pass










if __name__ == '__main__':
    # show_space_property(r"../../datasets/src_10")















