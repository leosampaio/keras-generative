import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import misc
from scipy.misc import imresize

import models

DATA_FOLDER = '/home/alex/datasets/bbc_pose/cropped'


def load_files():
    image_files = []
    for video_dir in os.listdir(DATA_FOLDER):
        cropped_path = os.path.join(DATA_FOLDER, video_dir)
        if os.path.isdir(cropped_path):
            for file in os.listdir(cropped_path):
                if file.endswith('_r.jpg'):
                    image_files.append(os.path.join(cropped_path, file))
    return image_files


def pad_img(img):
    dif_x = 64 - img.shape[0]
    dif_y = 64 - img.shape[1]

    if dif_x > 0 or dif_y > 0:
        if dif_y > 10 or dif_x > 10:
            print(dif_x, dif_y)
        img = np.pad(img, ((0, dif_x), (0, dif_y), (0, 0)), mode='constant')
    if img.shape[0] > 64 or img.shape[1] > 64:
        img = imresize(img, (64, 64, 3))

    return img


def load_image(image_path):
    image = misc.imread(image_path).astype(np.float32)
    image = pad_img(image)
    image = image / 127.5 - 1.
    return np.array(image)


def main():
    parser = argparse.ArgumentParser(description='clusters')
    parser.add_argument('--model', type=str, default='ALI')
    parser.add_argument('--weights', type=str,
                        default='/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc/ali/weights/epoch_00005')
    parser.add_argument('--z_dims', type=int, default=256)
    args = parser.parse_args()

    image_files = load_files()
    if len(image_files) < 1:
        print('No files found. Exiting.')
        exit()
    print(len(image_files))
    random.shuffle(image_files)

    max_samples = 50
    images = [load_image(image_file) for image_file in image_files[:max_samples]]
    images = np.array(images)
    print(images.shape)

    model = getattr(models, args.model)(
        input_shape=(64, 64, 3),
        z_dims=args.z_dims,
        output=''
    )
    model.load_model(args.weights)
    encodings = model.f_Gz.predict(images)
    encodings = encodings[0]
    reconstructions = model.predict_images(encodings)
    print(reconstructions.shape)

    images = images * 0.5 + 0.5
    images = np.clip(images, 0.0, 1.0)

    fig = plt.figure(figsize=(8, 8))
    grid = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)
    for i in range(0, 100, 2):
        ax = plt.Subplot(fig, grid[i])
        ax.imshow(images[i // 2, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, grid[i + 1])
        ax.imshow(reconstructions[i // 2, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)

    fig.savefig('sanity_check.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
