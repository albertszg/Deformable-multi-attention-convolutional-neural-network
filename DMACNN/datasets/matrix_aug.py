
import numpy as np
import torch

import random
from scipy.signal import resample
from PIL import Image
import PIL
import scipy

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        seq = seq[np.newaxis, :, :]
        return seq

class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)

class ReSize(object):
    def __init__(self, size=1):
        self.size = size
    def __call__(self, seq):
        # seq = scipy.misc.imresize(seq, self.size, interp='bilinear', mode=None)
        im = Image.fromarray(seq)
        size = tuple((np.array(im.size) * 1).astype(int))
        seq = np.array(im.resize(size, PIL.Image.BILINEAR))
        seq = seq / 255
        return seq


class AddGaussian(object):
    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)

class AddGaussian1(object):
    def __init__(self, snr=5):
        self.snr = snr

    def __call__(self, seq):
        noise = np.random.randn(seq.shape[0], seq.shape[1])  # 产生N(0,1)噪声数据
        noise = noise - np.mean(noise)  # 均值为0
        signal_power = np.linalg.norm(seq - seq.mean()) ** 2 / seq.size  # 此处是信号的std**2
        noise_variance = signal_power / np.power(10, (self.snr / 10))  # 此处是噪声的std**2
        noise = (np.sqrt(noise_variance) / np.std(noise)) * noise  ##此处是噪声的std**2
        signal_noise = noise + seq
        return signal_noise


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)

class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        return seq*scale_factor


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1, 1))
            return seq*scale_factor

class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_height = seq.shape[1] - self.crop_len
            max_length = seq.shape[2] - self.crop_len
            random_height = np.random.randint(max_height)
            random_length = np.random.randint(max_length)
            seq[random_length:random_length+self.crop_len, random_height:random_height+self.crop_len] = 0
            return seq

class Normalize(object):
    def __init__(self, type = "0-1"): # "0-1","1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if  self.type == "0-1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq