"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL.GimpGradientFile import EPSILON

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 323010835


# Reads an image and returns the image converted as requested (1 -> grayscale, 2 -> RGB)
def imReadAndConvert(filename:str, representation:int)->np.ndarray:
    image = cv2.imread(filename)
    # if grayscale
    if (representation == 1):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # normalize to [0,1]
        norm_image = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_image
    # if rgb
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # normalize to [0,1]
        norm_image = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_image

    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """



# Reads an image as RGB or grayscale and displays it
def imDisplay(filename:str, representation:int)->None:
    src = imReadAndConvert(filename, representation)
    cv2.imshow('im',src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """


# Converts RGB to YIQ
def transformRGB2YIQ(imRGB:np.ndarray)->np.ndarray:
    rgb_to_yiq = np.array([[0.299, 0.587, 0.114],
                           [0.59590059, -0.27455667, -0.32134392],
                           [0.21153661, -0.52273617, 0.31119955]])
    # Matrix multiply
    return np.dot(imRGB, rgb_to_yiq.T.copy())
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """


#Converts YIQ to RGB
def transformYIQ2RGB(imYIQ:np.ndarray)->np.ndarray:
    yiq_to_rgb = np.array([[0.299, 0.587, 0.114],
                           [0.59590059, -0.27455667, -0.32134392],
                           [0.21153661, -0.52273617, 0.31119955]])
    matrix = 1/yiq_to_rgb
    # Matrix multiply
    return np.dot(imYIQ, matrix.T.copy())
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """



# Histogram Equalization
# Returns the histograms and the equalized array
def histogramEqualize(imOrig:np.ndarray)->(np.ndarray,np.ndarray,np.ndarray):
    # Normalize the range back to [0,255]
    imOrig = imOrig*255
    image = imOrig.astype(np.uint8)
    # Show the image
    cv2.imshow('im', image)
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # CumSum and normalize it
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Show the histogram with CumSum
    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

    # Span the values according to the algorithm
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Transform the values to a 'better' image
    img2 = cdf[image]
    cv2.imshow('im', img2)
    hist, bins = np.histogram(img2.flatten(), 256, [0, 256])

    # CumSum of the new image (should be linear)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Show the histogram with CumSum
    plt.plot(cdf_normalized, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

    img2 = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img2, np.histogram(img2.flatten(), 256, [0, 256]), np.histogram(image.flatten(), 256, [0, 256])
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # return KmeansQuantizeImage(imOrig,nQuant,nIter)
    errors = list()
    images = list()
    is_rgb_image = len(imOrig.shape) == 3
    working_channel = imOrig
    if is_rgb_image:
        imOrig = transformRGB2YIQ(imOrig)
        working_channel = imOrig[:,:,0].copy()

    prob_w = np.bincount(working_channel.flatten(),minlength=256)/working_channel.size
    intes_count = np.cumsum(prob_w)
    borders_z = np.zeros((nQuant+1,),dtype=int)
    for i in range(nQuant-1):
        borders_z[i+1] = np.argmax(intes_count+EPSILON >= (i+1)/(nQuant+1))
    borders_z[-1] = 256
    centers_q = np.zeros((nQuant,),dtype=float)
    currImg = working_channel.copy()
    for k in range(nIter):
        print(borders_z,centers_q)
        for i in range(nQuant):
            vals = np.arange(borders_z[i], borders_z[i+1],dtype=float)
            border_prob = prob_w[borders_z[i]: borders_z[i + 1]]
            if border_prob.sum() == 0:
                centers_q[i] = 0
            else:
                centers_q[i] = (vals * border_prob).sum() / border_prob.sum()
        for i in range(0,nQuant-1):
            borders_z[i+1] = (centers_q[i]+centers_q[i+1])/2
        for i in range(nQuant):
            currImg[(borders_z[i] <= currImg) & (currImg < borders_z[i+1])] = int(centers_q[i])

        if is_rgb_image:
            imOrig[:, :, 0] = currImg
            print(np.max(currImg), np.min(currImg))
            images.append(transformYIQ2RGB(imOrig))
            print(np.max(images[-1]),np.min(images[-1]))
        else:
            images.append(currImg)
        curr_error = (np.sum(np.sqrt((working_channel.astype(float)-currImg.astype(float))**2)))/currImg.size
        errors.append(1.0/curr_error)
        if k!=0 and abs(errors[k-1]-errors[k])<EPSILON:
            break
    return images,errors




# --------------------------------------------------------------
def KmeansQuantizeImage(imOrig: np.ndarray, nQuant: int,nIter: int):
    errors = list()
    images = list()
    is_rgb_image = len(imOrig.shape) == 3
    working_channel = imOrig
    if is_rgb_image:
        # instead of convert to yiq
        working_channel = imOrig.reshape((-1, 3))
    colors = np.unique(imOrig.flatten())
    colors = colors.astype(int)
    centroids = colors[np.random.choice(colors.shape[0], nQuant + 1, replace=False)]
    for i in range(nIter):
        error = 0
        # Part 1
        clusters = []
        for j in range(nQuant):
            clusters.append([0])
        for intensity in colors:
            distances = np.zeros((nQuant,),dtype=float)
            n_times = float(np.count_nonzero(working_channel == intensity))
            for j in range(nQuant):
                curr_error = n_times * (np.linalg.norm(intensity - centroids[j]))
                distances[j] = curr_error
            clusters[distances.argmin()].append(intensity)
            error += min(distances)
        errors.append(error)
        # Part 2
        for j in range(nQuant):
            centroids[j] = np.round(np.average(clusters[j]))
            working_channel[np.in1d(working_channel,clusters[j]).reshape(working_channel.shape)] = centroids[j]

        if is_rgb_image:
            images.append(working_channel.reshape(imOrig.shape))
        else:
            images.append(working_channel)
        if i != 0 and errors[i-1]-errors[i] < EPSILON:
            break

    return images, errors

