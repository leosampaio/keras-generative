import numpy as np

from core.metrics import ImageSamplesMetric


class GeneratedSamples(ImageSamplesMetric):
    name = 'samples'
    input_type = 'generated_image_samples'

    def compute(self, input_data):
        imgs = input_data[:36]
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class ReconstructedSamples(ImageSamplesMetric):
    name = 'reconstruction'
    input_type = 'reconstruction_samples'

    def compute(self, input_data):
        originals, reconstructions = input_data

        imgs = np.zeros((len(originals)*2,) + originals.shape[1:])
        imgs[0::2] = originals
        imgs[1::2] = reconstructions
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs
