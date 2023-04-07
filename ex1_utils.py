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

    return np.dot(imgYIQ, np.linalg.inv(YIQKERNEL))



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


def hsitogramEqualize(imgOrig: np.ndarray) -> Tuple[np.ndarray]:
    """
    Equalizes the histogram of an image
    :param imgOrig: Original image
    :return: (imgEq,histOrg,histEQ)
    """ 
    
    color = len(imgOrig.shape) == 3
    imgEq = imgOrig.copy()

    if color:
        
        imgEq = transformRGB2YIQ(imgOrig)
        histEQ = cv.normalize(imgEq[:,:,0], None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F).astype(int).ravel()

    else: histEQ = imgEq.ravel()

    equalize(histEQ)

    if color:

        imgEq[:,:,0] = cv.normalize(histEQ.reshape(imgEq[:,:,0].shape), None, alpha = 0, beta = 1, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
        imgEq = transformYIQ2RGB(imgEq)

    else: imgEq = histEQ.reshape(imgEq.shape)


    return imgEq, hist(imgOrig[:,:,0].ravel() if color else imgOrig.ravel()), hist(histEQ)




def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> Tuple[List[np.ndarray], List[float]]:
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
