import os
import numpy as np
import cv2

IMAGE_SIZE = 64
images = []
labels = []

#==============================================================================
#将image统一补全并缩放成相同的大小
def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    #计算后续填充图片时的上、下、左、右的像素点
    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    #填充图片
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    #缩放图片以统一输入
    resized_image = cv2.resize(constant, (height, width))
    return resized_image
#==============================================================================

#==============================================================================
#读取图片，并返回大小统一的图片
def read_image(file_path):
    image = cv2.imread(file_path)
    #grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #灰度化图片 
    #image = cv.equalizeHist(grayImage) #直方图均衡化
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image
#==============================================================================

#==============================================================================
#遍历path，返回它下面的所有PNG图片及其所在路径
def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        #print(abs_path)
        if os.path.isdir(abs_path):  # 遍历子目录
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.png'):
                image = read_image(abs_path)
                images.append(image)
                labels.append(path)

    return images, labels
#==============================================================================

#==============================================================================
#给获得的图片贴上标签
def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    labels = np.array([0 if label.endswith('me') else 1 for label in labels])

    return images, labels
#==============================================================================



