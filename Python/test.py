from  ISRS import getsuperpixle
import numpy as np
import cv2,os
import time
import scipy.io as io
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import thin
import matplotlib.pyplot as plt


def boundary(label):
    kernel_horizontal = np.array([[0, -1, 1]], dtype=np.int32)
    grad_horizontal = np.abs(convolve(label, kernel_horizontal, mode='nearest'))

    # 垂直方向梯度
    kernel_vertical = np.array([[0], [-1], [1]], dtype=np.int32)
    grad_vertical = np.abs(convolve(label, kernel_vertical, mode='nearest'))

    # 合并边界
    boundaries = (grad_horizontal > 0) | (grad_vertical > 0)

    # 细化边界
    thinned_boundaries = thin(boundaries)
    return thinned_boundaries
image = cv2.imread(r"F:\Data\BSDS500\Clean\images\2018.jpg").astype(np.int32)

ori_label = getsuperpixle(image, 100)
thin_boundary = boundary(ori_label)

image[thin_boundary] = [0, 0, 255]
cv2.imwrite("2018_ISRS.jpg",image)