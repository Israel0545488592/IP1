from typing import List, Tuple

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
YIQKERNEL = np.array([[0.299,  0.587,  0.114 ],
                      [0.596, -0.275, -0.321 ], 
                      [0.212, -0.523,  0.311 ]])



def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 212403679


def discrete_normalize(arr: np.ndarray) -> np.ndarray:
    return cv.normalize(arr, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F).astype(np.uint8)

def continuous_normalize(arr: np.ndarray) -> np.ndarray:
    return cv.normalize(arr, None, alpha = 0, beta = 1, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    return continuous_normalize(cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB if representation == LOAD_RGB else cv.COLOR_BGR2GRAY))


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    plt.title('RGB' if representation == LOAD_RGB else 'GRAY_SCALE')
    plt.imshow(imReadAndConvert(filename, representation), cmap = 'gray' if representation == LOAD_GRAY_SCALE else None)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    return np.dot(imgRGB, YIQKERNEL.transpose())


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    return np.dot(imgYIQ, np.linalg.inv(YIQKERNEL).transpose())


def hist(arr: np.ndarray) -> np.ndarray:
    """ PDF """

    return np.histogram(arr, bins = 256)[0]


def hist_equalized(img: np.ndarray) -> np.ndarray:
    """ histogram equalization """

    lin_cdf = discrete_normalize(np.cumsum(hist(img.ravel())) / img.size)

    return np.vectorize(lambda col : lin_cdf[col]) (img)


def hsitogramEqualize(imgOrig: np.ndarray) -> Tuple[np.ndarray]:
    """
    Equalizes the histogram of an image
    :param imgOrig: Original image
    :return: (imgEq,histOrg,histEQ)
    """ 
    color = len(imgOrig.shape) == 3

    img = transformRGB2YIQ(imgOrig.copy()) if color else imgOrig.copy()
    imgEq = hist_equalized(discrete_normalize(img[:, :, 0] if color else img))
    histEQ = hist(imgEq.ravel())

    if color: imgEq = transformYIQ2RGB(np.dstack((continuous_normalize(imgEq), img[:, :, 1:])))

    return imgEq, hist(discrete_normalize(img[:, :, 0] if color else img).ravel()), histEQ


def expectation(pdf, start, end):
    return int(np.arange(start, end) @ pdf[start:end] / max(sum(pdf[start:end]), 1))

def MSE(pdf, bounds, centroids):
    return np.sqrt(sum((val - centroids[ind]) ** 2  for ind in range(len(bounds) - 1) for val in pdf[bounds[ind] : bounds[ind - 1]])) / sum(pdf)


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> Tuple[List[np.ndarray], List[float]]:
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    color = len(imOrig.shape) == 3
    img_history, error_history = [], []

    base_img = discrete_normalize(transformRGB2YIQ(imOrig)[:, :, 0] if color else imOrig)

    pdf = hist(base_img.ravel())
    bounds = list(np.linspace(0, 255, nQuant + 1).astype(int))
    centroids = np.ones(nQuant)

    for _ in range(nIter):

        img = base_img.copy()

        for ind in range(nQuant): centroids[ind] = expectation(pdf, bounds[ind], bounds[ind + 1])

        for ind in range(1, nQuant): bounds[ind] = int((centroids[ind - 1] + centroids[ind]) / 2)
    
        img = np.vectorize(lambda col : centroids[int(col * nQuant / 256)]) (img)

        if color:   img = transformYIQ2RGB(np.dstack((continuous_normalize(img), transformRGB2YIQ(imOrig)[:, :, 1:])))

        img_history.append(img)
        error_history.append(MSE(pdf, bounds, centroids))


    return img_history, error_history