import cv2 as cv
from CNN_train import Model
#from image_show import show_image
face_cascade = cv.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    model = Model()
    model.load()
    while cap.isOpened():
        _, image = cap.read()
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #灰度化图片 
        equalImage = cv.equalizeHist(grayImage) #直方图均衡化
        faces = face_cascade.detectMultiScale(equalImage, scaleFactor=1.3, minNeighbors=3)
        if len(faces) > 0:
            print('face detected')
            color = (255, 255, 255)  # 白
            for (x,y,w,h) in faces:
                #裁剪出人脸，单独保存成图片
                head = image[y-10:y+h,x:x+w]

                result = model.predict(head)
                if result == 0:  # boss
                    print('Joey,你好！')
                    #show_image()
                else:
                    print('......')

        key = cv.waitKey(40)
        if key == 27 or key == ord('q'): #如果按ESC或q键，退出
            break

    cap.release()
    cv.destroyAllWindows()
