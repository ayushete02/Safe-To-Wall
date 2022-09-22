import random
import cv2
import datetime
import numpy as np
from PIL import Image
import time

scr = cv2.VideoCapture(0)
# faceCascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
pointer = -320
pos = -319

while True:
    ret, frame = scr.read()
    frame = cv2.flip(frame, 1)
    j = 51
    submarine = cv2.imread('submarine.png')
    height = 30
    width = 60
    submarine = cv2.resize(submarine, (width,height))
    # img2gray = cv2.cvtColor(submarine, cv2.COqLOR_BGR2GRAY)
    ret, mask = cv2.threshold(submarine, 1, 255, cv2.THRESH_BINARY)
    roi0 = frame[-height-300:-300, -width-380:-380]

    roi0[np.where(mask)] = 0
    roi0 += submarine
    cv2.putText(frame,'-',(255, 178),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),0)
    cv2.putText(frame,'-',(195, 178),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),0)
    cv2.putText(frame,'-',(225, 160),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),0)
    cv2.putText(frame,'-',(225, 188),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),0)
    cv2.putText(frame,'-',(247, 186),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),0)
    cv2.putText(frame,'-',(245, 170),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),0)
    cv2.putText(frame,'-',(207, 186),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),0)
    cv2.putText(frame,'-',(205, 170),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),0)



    
    if pointer<-240 and pointer>-320:
        cv2.putText(frame,'3',(100, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
    if pointer<-160 and pointer>-240:
        cv2.putText(frame,'2',(100, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
    if pointer<-80 and pointer>-160:
        cv2.putText(frame,'1',(100, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
    if pointer<-0 and pointer>-80:
        cv2.putText(frame,'Go',(100, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)

    h = random.randint(0, 10)
    if pos >= 0:
        i = pointer % 640
        if i == 0:
            h1 = h
            i = 1
        print(h1)
        emoj = cv2.imread(f'{h1}.png')
        emoj = cv2.resize(emoj, (40, 478))
        img2gray = cv2.cvtColor(emoj, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

        if i < 600:
            # cv2.putText(frame,f'.',(i, h1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
            roi1 = frame[-478-1:-1, -40-i:-i]

            roi1[np.where(mask)] = 0
            roi1 += emoj

    h = random.randint(0, 10)
    if pos > 160:
        j = i - 160
        if j == 0:
            h2 = h
            j = 1
        if j < 0:
            j = 640+j
        if j < 600:
            print(j)
            emoj = cv2.imread(f'{h2}.png')

            emoj = cv2.resize(emoj, (40, 478))
            img2gray = cv2.cvtColor(emoj, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        # cv2.putText(frame,f'.',(i, h1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
            roi2 = frame[-478-1:-1, -40-j:-j]

            roi2[np.where(mask)] = 0
            roi2 += emoj

    h = random.randint(0, 10)
    if pos > 320:
        k = j - 160
        if k == 0:
            h3 = h
            k = 1
        if k < 0:
            k = 640+k
        if k < 600:
            emoj = cv2.imread(f'{h3}.png')

            emoj = cv2.resize(emoj, (40, 478))
            img2gray = cv2.cvtColor(emoj, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            # cv2.putText(frame,f'.',(i, h1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
            roi3 = frame[-478-1:-1, -40-k:-k]

            roi3[np.where(mask)] = 0
            roi3 += emoj

    h = random.randint(0, 10)
    if pos > 480:
        l = k - 160
        if l == 0:
            h4 = h
            l = 1
        if l < 0:
            l = 640+l
        if l < 600:
            emoj = cv2.imread(f'{h4}.png')

            emoj = cv2.resize(emoj, (40, 478))
            img2gray = cv2.cvtColor(emoj, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            # cv2.putText(frame,f'.',(i, h1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
            roi4 = frame[-478-1:-1, -40-l:-l]

            roi4[np.where(mask)] = 0
            roi4 += emoj

    pointer = pointer+2

    pos = pos+2
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break


scr.release()

cv2.destroyAllWindows()
