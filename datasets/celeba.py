import os
import requests
import glob
import argparse

import numpy as np
import scipy as sp
import imageio
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, default='datasets/files')
args, _ = parser.parse_known_args()

url = "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1"
outdir = args.data_folder
h5_filepath = os.path.join(outdir, 'celeba{}.h5')
out_zipfile = os.path.join(outdir, 'celeba.zip')
out_unzipped_folder = os.path.join(outdir, 'unzipped_celeba')

CHUNK_SIZE = 32768
DISK_BUFFER_SIZE = 30000
IMG_SIZE = 64


def download_celeba():
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    session = requests.Session()
    response = session.get(url, stream=True)
    with open(out_zipfile, 'wb') as fp:
        dl = 0
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                dl += len(chunk)
                fp.write(chunk)
                mb = dl / 1.0e6
                print('{:.2f} MB downloaded...'.format(mb), end='\r', flush=True)
        print('Done!')


def preprocess_image(x, crop_size=IMG_SIZE):
    smallest_size = np.min((x.shape[0], x.shape[1]))
    crop_y = (x.shape[0] - smallest_size) // 2
    crop_x = (x.shape[1] - smallest_size) // 2
    x = x[crop_y:(x.shape[0] - crop_y), crop_x:(x.shape[1] - crop_x), :]
    x = x.astype(np.float32)
    return sp.misc.imresize(x, (crop_size, crop_size)) / 255.


def load_data(image_size=64):
    """
    Load and return dataset as tuple (data, label, label_strings)
    """

    if not os.path.exists(out_zipfile):
        download_celeba()
    print("Downloaded celeba!")

    if not os.path.exists(out_unzipped_folder):
        import zipfile
        with zipfile.ZipFile(out_zipfile, 'r') as zip_ref:
            print("Unzipping contents... ", end='')
            zip_ref.extractall(out_unzipped_folder)
            print("Done!")

    h5_f_filepath = h5_filepath.format(image_size)
    if not os.path.exists(h5_f_filepath):
        image_paths = glob.glob("{}/img_align_celeba/*.jpg".format(out_unzipped_folder))
        data_lenght = len(image_paths)
        count = 0
        with h5py.File(h5_f_filepath, mode='w') as hdf5_file:
            hdf5_file.create_dataset("data", (data_lenght, image_size, image_size, 3), np.float32)
            for img_path in image_paths:
                img = preprocess_image(imageio.imread(img_path), crop_size=image_size)
                hdf5_file['data'][count, ...] = img
                count += 1
                print('Preprocessed', count, 'images', end='\r')
    return h5_f_filepath

if __name__ == '__main__':
    datapath = load_data()
