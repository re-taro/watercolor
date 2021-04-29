import cv2
import numpy as np
import os.path
import matplotlib as plt
from PIL import Image

CONTENT = input('content:') 
CONTENT = 'INPUT/' + CONTENT

#####初期設定#############
average_square = (5,5)
sigma_x = 0
reshape_size = (-1,3)
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
k = 100
line_average = (9,9)
line_sigma_x = 0
threshold1 = 50
threshold2 = 55
edges_average = (1,1)
edges_sigma_x = 0
thresh = 90
max_pixel = 255
gamma = 5.0
multi_w = 0.5
paint_w = 0.9
gamma = 1.5


image = cv2.imread(CONTENT,1)
file, ext = os.path.splitext(CONTENT)

def paint(filename):
    image_blurring = cv2.GaussianBlur(filename, average_square, sigma_x)
    z = image_blurring.reshape(reshape_size)
    z = np.float32(z)
    ret, label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    image_reshape = res.reshape((image_blurring.shape))
    return cv2.GaussianBlur(image_reshape, average_square, sigma_x)

def line(filename):
    image_processed = cv2.cvtColor(cv2.GaussianBlur(filename, line_average, line_sigma_x), cv2.COLOR_BGR2GRAY)
    image_edges = cv2.Canny(image_processed, threshold1 = threshold1, threshold2 = threshold2)
    image_h = cv2.GaussianBlur(image_edges, edges_average, edges_sigma_x)
    _, image_binary = cv2.threshold(image_h, thresh, max_pixel, cv2.THRESH_BINARY)
    image_binary = cv2.bitwise_not(image_binary)
    return image_binary

paint = paint(image)
cv2.imwrite(file + '_paint' + ext, paint)
line = line(image)
cv2.imwrite(file + '_line' + ext, line)

def mul(input_color, mul_color):
    return int(round(((input_color * mul_color)/255),0))

def multiple(image1, image2):
    pixelSizeTuple = image1.size
    image3 = Image.new('RGB', image1.size)
    for i in range(pixelSizeTuple[0]):
        for j in range(pixelSizeTuple[1]):
            r, g, b = image1.getpixel((i, j))
            r2, g2, b2 = image2.getpixel((i, j))
            img_r = mul(r, r2)
            img_g = mul(g, g2)
            img_b = mul(b, b2)
            image3.putpixel((i, j), (img_r ,img_g, img_b))
    return image3

image1 = Image.open(file + '_paint' + ext).convert('RGB')
image2 = Image.open(file + '_line' + ext).convert('RGB')

multi = multiple(image1, image2)
multi.save(file + '_multi'+ ext)

def dodge(multi, paint):
    d = cv2.addWeighted(multi, multi_w, paint, paint_w, gamma)
    return d

multi_image = cv2.imread(file + '_multi' + ext, 1)
output = dodge(multi_image, paint)

cv2.imwrite(file + '_output' + ext, output)