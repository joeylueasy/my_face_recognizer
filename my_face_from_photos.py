import cv2 as cv
#from PIL import Image
import numpy as np
import time
import os


face_cascade = cv.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
face_num = 0
frame_num = 0
root_path = '' 
#==============================================================================
#删除当前目录下所有的.*文件
def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith("png") or name.endswith("avi"):
                os.remove(os.path.join(root, name))
#==============================================================================

#==============================================================================
def rename(path):
    global frame_num
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith("jpg") or name.endswith("JPG"):
                os.rename(os.path.join(os.path.join(root_path,'photos'),name),
                          os.path.join(os.path.join(root_path,'photos'),"pic_%s.jpg"%(frame_num)))
                frame_num = frame_num + 1
                
#==============================================================================
              
#==============================================================================
def get_my_face(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith("jpg"):
                img = cv.imread(os.path.join(root, name)) #打开该照片
                print('now is %s' % name)
                face_detector(img, face_cascade)
                
#==============================================================================
                
#==============================================================================
# 检测出图片中的人脸，并用方框标记出来
def face_detector(image, cascade):
    global face_num
    global root_path
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #灰度化图片 
    equalImage = cv.equalizeHist(grayImage) #直方图均衡化
    faces = cascade.detectMultiScale(equalImage, scaleFactor=1.3, minNeighbors=3)
        
    os.chdir(os.path.join(root_path,'training'))
    for (x,y,w,h) in faces:
        #裁剪出人脸，单独保存成图片，注意这里的横坐标与纵坐标不知为啥颠倒了
        cv.imwrite("face_%s.png"%(face_num), image[y:y+h,x:x+w])
        face_num = face_num + 1
        
#==============================================================================

               
if __name__ == '__main__':   
    root_path = os.getcwd()
    del_files(os.path.join(root_path,'training'))
    #rename(os.path.join(root_path,'photos'))
    get_my_face(os.path.join(root_path,'photos'))
    print('done')
