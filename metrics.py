import cv2
import pywt
import skimage.measure
import imquality.brisque as brisque
import webcolors as wc
from skimage.filters import sobel
from pybdm import BDM
import pylab as pl
import numpy as np
import sys

# The cpbd module tries to import scipy.ndimage.imread, which does not exist. This line below is a fix for this issue.
# noinspection PyTypeChecker
sys.modules['scipy.ndimage.imread'] = cv2.imread
import cpbd

# TODO: check for RGB/BGR inconsistencies
# TODO: validate metrics (is the implementation correct?)
# TODO: check correlation between k_complexity_bw() and k_complexity_lightness() (better yet, check all correlations)
# TODO: wrt to the k_complexity functions, is the metric of 2D array sufficiently correlated with its 1D version?

# TODO: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2011.01900.x?casa_token=VU-te9nefM8AAAAA%3AqcRGedXRra5yodIDCXAUdS3m1mHCIyxaui2IsbWiB2Iq2S3xNnRs1xb6GMf7um2foQNzBIxFrg8rdTM
# TODO: https://ieeexplore.ieee.org/document/5995467


def contrast_rms(image):
    # Returns RMS contrast
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.std()


# noinspection PyTypeChecker
def contrast_tenengrad(image):
    # Returns tenengrad contrast
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_img = sobel(image) ** 2
    return np.sqrt(np.sum(sobel_img)) / image.size * 10000


def fractal_dimension(image):
    # Adapted from https://francescoturci.net/2016/03/31/box-counting-in-numpy/
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # finding all the non-zero pixels
    pixels = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > 0:
                pixels.append((i, j))

    lx = image.shape[1]
    ly = image.shape[0]
    pixels = np.array(pixels)

    # computing the fractal dimension
    # considering only scales in a logarithmic list
    scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
    ns = []
    # looping over several scales
    for scale in scales:
        # computing the histogram
        h, edges = np.histogramdd(pixels, bins=(np.arange(0, lx, scale), np.arange(0, ly, scale)))
        ns.append(np.sum(h > 0))

    # linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(scales), np.log(ns), 1)

    return -coeffs[0]  # the fractal dimension is the OPPOSITE of the fitting coefficient


def sharpness(image):
    # https://pypi.org/project/cpbd/
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cpbd.compute(image)


def sharpness_laplacian(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()


def brisque_score(image):
    return brisque.score(image)


def color_dominant(image):

    def get_approx_color(hex_color):
        orig = wc.hex_to_rgb(hex_color)
        similarity = {}
        for hex_code, color_name in wc.CSS3_HEX_TO_NAMES.items():
            approx = wc.hex_to_rgb(hex_code)
            similarity[color_name] = sum(np.subtract(orig, approx) ** 2)
        return min(similarity, key=similarity.get)

    def get_color_name(hex_color):
        try:
            return wc.hex_to_name(hex_color)
        except ValueError:
            return get_approx_color(hex_color)

    # https://stackoverflow.com/questions/50899692/most-dominant-color-in-rgb-image-opencv-numpy-python
    # https://stackoverflow.com/questions/44354437/classify-users-by-colors
    data = np.reshape(image, (-1, 3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, 1, None, criteria, 10, flags)
    bgr_ = centers[0].astype(np.int32)
    rgb_ = bgr_
    rgb_[0], rgb_[2] = bgr_[2], bgr_[0]  # convert BGR to RGB
    hex_ = wc.rgb_to_hex(tuple(rgb_))

    return get_color_name(hex_)


def colorfulness(image):
    # https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
    # "Measuring colourfulness in natural images" David Hasler and Sabine Susstrunk
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype('float'))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    std_root = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    mean_root = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return std_root + (0.3 * mean_root)

# First order


def pixel_intensity_mean(image):
    return image.mean()


def hue_mean(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV).mean()


def saturation_mean(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1].mean()
    return saturation


# Second order

# Higher order
def entropy_shannon(image):
    return skimage.measure.shannon_entropy(image)


from numba import jit


# @jit(nopython=True, cache=True)
@jit
def k_complexity_bw(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(image.shape[0]*image.shape[1])
    image = image*(252/256)
    bins = list(range(0, 252, 28))
    image = np.digitize(image, bins=bins)-1
    bdm = BDM(ndim=1, nsymbols=9, warn_if_missing_ctm=False)
    return bdm.bdm(image)


def k_complexity_lab_l(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image = image.astype('float32') / 255  # transformation to fix Euclidian distances in Lab space
    image = image[:, :, 0]
    image = image.reshape(image.shape[0]*image.shape[1])
    bins = list(np.arange(0, 0.9, 0.1))
    image = np.digitize(image, bins=bins)-1
    bdm = BDM(ndim=1, nsymbols=9, warn_if_missing_ctm=False)
    return bdm.bdm(image)


def k_complexity_lab_a(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image = image.astype('float32') / 255  # transformation to fix Euclidian distances in Lab space
    image = image[:, :, 1]
    image = image.reshape(image.shape[0]*image.shape[1])
    bins = list(np.arange(0, 0.9, 0.1))
    image = np.digitize(image, bins=bins)-1
    bdm = BDM(ndim=1, nsymbols=9, warn_if_missing_ctm=False)
    return bdm.bdm(image)


def k_complexity_lab_b(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image = image.astype('float32') / 255  # transformation to fix Euclidian distances in Lab space
    image = image[:, :, 2]
    image = image.reshape(image.shape[0]*image.shape[1])
    bins = list(np.arange(0, 0.9, 0.1))
    image = np.digitize(image, bins=bins)-1
    bdm = BDM(ndim=1, nsymbols=9, warn_if_missing_ctm=False)
    return bdm.bdm(image)


def haar_wavelet(image):
    image = image[::2, ::2]  # resize by factor 2
    return pywt.dwt2(image, 'Haar')
