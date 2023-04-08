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


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation == 1: return cv.imread(filename, cv.IMREAD_GRAYSCALE)
    return cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    plt.imshow(imReadAndConvert(filename, representation), cmap = 'gray' if representation == 1 else None)
    plt.grid(False)
    plt.axis(False)


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

    hist = np.zeros(256).astype(int)
    for pix in arr: hist[pix] += 1
    return hist


def cum_hist(arr: np.ndarray) -> np.ndarray:
    """ CDF """

    cum_sum = np.zeros_like(arr)
    cum_sum[0] = arr[0]
    for i in range(1, len(arr)): cum_sum[i] = cum_sum[i-1] + arr[i] 
    return cum_sum


def equalize(img: np.ndarray) -> None:
    """ histogram equalization """

    lin_cdf = ((cum_hist(hist(img)) / img.size) * 255).astype(int)

    for ind in range(len(img)): img[ind] = lin_cdf[img[ind]]


''' The folowing 2 methods are for processing the Y - channle of a YIQ formatted image '''

def descret_luminance(luminance: np.ndarray) -> np.ndarray:
    return cv.normalize(luminance, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F).astype(int)

def continuous_luminance(luluminance: np.ndarray) -> np.ndarray:
    return cv.normalize(luluminance, None, alpha = 0, beta = 1, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)


def hsitogramEqualize(imgOrig: np.ndarray) -> Tuple[np.ndarray]:
    """
    Equalizes the histogram of an image
    :param imgOrig: Original image
    :return: (imgEq,histOrg,histEQ)
    """ 
    
    color = len(imgOrig.shape) == 3

    imgEq = transformRGB2YIQ(imgOrig.copy()) if color else imgOrig.copy()

    histEQ = descret_luminance(imgEq[:,:,0]).ravel() if color else imgEq.ravel()

    equalize(histEQ)

    if color:

        imgEq[:,:,0] = continuous_luminance(histEQ).reshape(imgEq[:,:,0].shape)
        imgEq = transformYIQ2RGB(imgEq)

    else: imgEq = histEQ.reshape(imgEq.shape)


    return imgEq, hist(imgOrig[:,:,0].ravel() if color else imgOrig.ravel()), hist(histEQ)



def expectation(pdf, start, end):
    return int(np.arange(start, end) @ pdf[start:end] / max(sum(pdf[start:end]), 1))

def MSE(pdf, bounds, centroids):
    return sum((val - centroids[ind]) ** 2  for ind in range(len(bounds) - 1) for val in pdf[bounds[ind] : bounds[ind - 1]]) / sum(pdf)


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

    if color:
        imOrig = transformRGB2YIQ(imOrig)
        imOrig[:,:,0] = descret_luminance(imOrig[:,:,0])

    pdf = hist(imOrig[:,:,0].ravel() if color else imOrig.ravel())

    bounds = list(np.linspace(0, 255, nQuant + 1).astype(int))
    centroids = np.ones(nQuant)

    for _ in range(nIter):

        img = imOrig.copy()

        for ind in range(nQuant): centroids[ind] = expectation(pdf, bounds[ind], bounds[ind + 1])

        for ind in range(1, nQuant): bounds[ind] = int((centroids[ind - 1] + centroids[ind]) / 2)
    
        quantizer = lambda col : centroids[int(col * nQuant / 256)]

        for row in (img[:,:,0] if color else img):
            for ind in range(len(row)): row[ind] = quantizer(row[ind])

        if color:
            img[:,:,0] = continuous_luminance(img[:,:,0])
            img = transformYIQ2RGB(img)

        img_history.append(img)
        error_history.append(MSE(pdf, bounds, centroids))


    return img_history, error_history