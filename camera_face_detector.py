import cv2 as cv
#from PIL import Image
import numpy as np
import time
import os

face_cascade = cv.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
face_num = 0
frame_num = 0
#==============================================================================
#删除当前目录下所有的.*文件
def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            #print(os.path.join(root, name))
            if name.endswith("png") or name.endswith("avi"):
                os.remove(os.path.join(root, name))
#==============================================================================

#==============================================================================
# 检测出图片中的人脸，并用方框标记出来
def face_detector(image, cascade):
    global face_num #引用全局变量
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #灰度化图片 
    equalImage = cv.equalizeHist(grayImage) #直方图均衡化
    faces = cascade.detectMultiScale(equalImage, scaleFactor=1.3, minNeighbors=3)
        
    for (x,y,w,h) in faces:
        #裁剪出人脸，单独保存成图片，注意这里的横坐标与纵坐标不知为啥颠倒了
        #cv.imwrite("face_%s.png" %(face_num), image[y:y+h,x:x+w])
        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        face_num = face_num + 1
    return image
#==============================================================================

#==============================================================================
#打开并显示图片
#img = cv.imread("IMG_2009.jpg") #打开一张图片
#show_image = face_detector(img, face_cascade)
##cv.imshow("Image", show_image) #在窗口中显示图片
#cv.waitKey(0) #无限期地等待键盘输入
#cv.destroyAllWindows() #释放窗口
#==============================================================================

#==============================================================================
#打开摄像头，循环显示一帧
#创建摄像头对象，其参数0表示第一个摄像头，一般就是笔记本的内建摄像头
#cap = cv.VideoCapture(0)
#cv.namedWindow("camera") #创建一个窗口
#while cap.isOpened():
#    ret, frame = cap.read() #读取一帧
#    cv.imshow("camera", frame) #显示一帧
#    key = cv.waitKey(10)
#    if (key & 0xFF == ord('q')) | (key == 27):
#        break
#cap.release()
#cv.destroyAllWindows()
#==============================================================================

#==============================================================================
def camera_detector():
    global frame_num
    #del_files(os.getcwd()) #每次先删除之前存储的png截图和视频
    cap = cv.VideoCapture(0) #打开笔记本内置的摄像头
    
#    videoName = time.strftime('%Y-%m-%d_%H_%M')+".avi"
#    fourcc = cv.VideoWriter_fourcc(*'XVID') #定义视频编码方式
#    out = cv.VideoWriter(videoName, fourcc, 24.0, (640,480)) #创建视频对象
    
    
    while cap.isOpened():
        #time.sleep(1) #延迟x秒
        ret, frame = cap.read() #读取一帧，前一个返回值是是否成功，后一个返回值是图像本身
#        out.write(frame) #把每帧图片一帧帧地写入video中
        show_image = face_detector(frame, face_cascade) #show_image是返回的已标记出人脸的图片
        cv.imshow("monitor", show_image) #在窗口显示一帧
        
        key = cv.waitKey(40)
        if key == 27 or key == ord('q'): #如果按ESC或q键，退出
            break
        if key == ord('s'): #如果按s键，保存图片
            cv.imwrite("frame_%s.png" % frame_num,frame)
            frame_num = frame_num+1
#    out.release()
    cap.release()
    cv.destroyAllWindows()
#==============================================================================

#当该py文件被直接运行时，代码将被运行，当该py文件是被导入时，代码不被运行   
if __name__ == '__main__':   
    camera_detector()



    