from numpy.lib.function_base import kaiser
from numpy.lib.npyio import savez_compressed
import handTrackingModule as htm
import time
from PIL import Image, ImageDraw
import random
import os
import cv2
import numpy as np
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0


wcam, hcam = 640, 480
print("Opening Camera")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Camera Opened")
# cap.set(3, wcam)
# cap.set(4, hcam)
pTime = 0
detector = htm.handDetector(detectionCon=0.75)

pointer = -320
pos = -319
    

LastPx = 0
LastPy = 0

speed = 4


# def Crash(score):
#     cv2.rectangle(frame, (200, 150), (440, 280), (0, 0, 0), -1)
#     cv2.putText(frame, 'GAME OVER', (230, 200),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # 1 wall up
#     cv2.putText(frame, f'score : {score}', (235, 240),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)  # 1 wall up
#     input()


score = 0
GameOver = 0
GameOverNotify = 0
while True:
    print(GameOver)
    if GameOver == 1:
        input()
    GameOver = 0
    GameOverNotify = 0

    x1 = LastPx
    y1 = LastPy
    # print(x1, y1)

    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame, draw=True)
    lmList = detector.findPosition(frame, draw=False)
    # print(lmList)
    tipId = [4, 8, 12, 16, 20]
    if(len(lmList) != 0):
        fingers = []
        # thumb
        if(lmList[tipId[0]][1] > lmList[tipId[0]-1][1]):
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 fingers
        for id in range(1, len(tipId)):

            if(lmList[tipId[id]][2] < lmList[tipId[id]-2][2]):
                fingers.append(1)

            else:
                fingers.append(0)
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, 1080))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, 720))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
            y1 = 480-y1
            x1 = 640 - x1

    submarine = cv2.imread('submarine.png')
    height = 30
    width = 60
    submarine = cv2.resize(submarine, (width, height))
    # img2gray = cv2.cvtColor(submarine, cv2.COqLOR_BGR2GRAY)
    ret, mask = cv2.threshold(submarine, 1, 255, cv2.THRESH_BINARY)
    # print(x1, y1)

    # print(y1)
    if y1 > 450:
        y1 = 450
    if y1 < 40:
        y1 = 40
    if x1 > 580:
        x1 = 580
    if x1 < 60:
        x1 = 60
    # roi0 = frame[-height-y1:-y1, -width-x1:-x1]
    roi0 = frame[-height-y1:-y1, -width-380:-380]

    roi0[np.where(mask)] = 0
    roi0 += submarine

    # cv2.putText(frame,'_',(640-380,470-y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),1)   #front
    # cv2.putText(frame,'_',(640-440,470-y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),1)   #back

    # cv2.putText(frame,'_',(640-410,480-y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),1) #f1up
    # cv2.putText(frame,'_',(640-410,450-y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),1) #f1down

    # cv2.putText(frame,'_',(640-395,480-y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0,255),1)  #m1up
    # cv2.putText(frame,'_',(640-395,460-y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),1) #m1down

    # cv2.putText(frame,'_',(640-425,480-y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),1)   #b1up
    # cv2.putText(frame,'_',(640-425,455-y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),1)   #b1down

    submarine_coordinate = []
    # middleline
    for count in (380, 440):
        submarine_coordinate.append((640-count, 470-y1))
        submarine_coordinate.append((640-count, 465-y1))
        submarine_coordinate.append((640-count, 475-y1))

    # outer point
    # submarine_coordinate.append((640-380,470-y1))
    # submarine_coordinate.append((640-440,470-y1))
    submarine_coordinate.append((640-410, 480-y1))
    submarine_coordinate.append((640-410, 450-y1))
    submarine_coordinate.append((640-395, 480-y1))
    submarine_coordinate.append((640-395, 460-y1))
    submarine_coordinate.append((640-425, 480-y1))
    submarine_coordinate.append((640-425, 455-y1))
    # print(submarine_coordinate)

    if pointer < -240 and pointer > -320:
        cv2.putText(frame, '3', (200, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 10)
    if pointer < -160 and pointer > -240:
        cv2.putText(frame, '2', (200, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 10)
    if pointer < -80 and pointer > -160:
        cv2.putText(frame, '1', (200, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 10)
    if pointer < -0 and pointer > -80:
        cv2.putText(frame, 'Go', (100, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 10)

    h = random.randint(0, 10)
    if pos >= 0:
        i = pointer % 640
        if i == 0:
            h1 = h
            i = 1
        # print(h1)
        emoj = cv2.imread(f'{h1}.png')
        emoj = cv2.resize(emoj, (40, 478))
        img2gray = cv2.cvtColor(emoj, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

        if i < 600:
            # cv2.putText(frame,f'.',(i, h1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
            roi1 = frame[-478-1:-1, -40-i:-i]

            roi1[np.where(mask)] = 0
            roi1 += emoj
            if i == 380:
                score = score + 1

        # cv2.putText(frame,'_',(640-i,145+h1*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)         #1 wall up
        # cv2.putText(frame,'_',(640-i-20,145+h1*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #2 wall up
        # cv2.putText(frame,'_',(640-i-40,145+h1*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #3 wall up

        # cv2.putText(frame,'_',(640-i,145+h1*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #1 wall down
        # cv2.putText(frame,'_',(640-i-20,145+h1*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)   #2 wall down
        # cv2.putText(frame,'_',(640-i-40,145+h1*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)   #3 wall down

        # cv2.line(frame,(640-i-40,0),(640-i-40,145+h1*10-1),(0, 0, 255),1)        #1 wall
        # cv2.line(frame,(640-i-40,480),(640-i-40,145+h1*10+70-3),(0, 0, 255),1)   #2 wall

        wallone = []
        for count in range(0, 40):
            wallone.append((640-i-count, 145+h1*10-1))
            wallone.append((640-i-count, 145+h1*10+70-3))

        for item1 in range(145+h1*10-1):
            wallone.append((640-i-40, item1))

        for item2 in range(145+h1*10+70-3, 480):
            wallone.append((640-i-40, item2))

        if set(submarine_coordinate).intersection(wallone) != set():
            cordinate_list = set(submarine_coordinate).intersection(wallone)
            print(cordinate_list)
            cordinate_list = list(cordinate_list)
            # x = cordinate_list[0][0]
            # y = cordinate_list[0][1]
            # # Crash(cordinate_list[0][0],cordinate_list[0][1])
            # emoj = cv2.imread('boom.png')
            # emoj = cv2.resize(emoj, (40, 32))
            # img2gray = cv2.cvtColor(emoj, cv2.COLOR_BGR2GRAY)
            # ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

            # roi1 = frame[-32-1:-1, -40-1:-1]

            # roi1[np.where(mask)] = 0
            # roi1 += emoj
            # Crash(score)
            GameOverNotify = 999

    h = random.randint(0, 10)
    if pos > 160:
        j = i - 160
        if j == 0:
            h2 = h
            j = 1
        if j < 0:
            j = 640+j
        if j < 600:
            if j == 380:
                score = score + 1
            # print(j)
            emoj = cv2.imread(f'{h2}.png')

            emoj = cv2.resize(emoj, (40, 478))
            img2gray = cv2.cvtColor(emoj, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        # cv2.putText(frame,f'.',(i, h1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
            roi2 = frame[-478-1:-1, -40-j:-j]

            roi2[np.where(mask)] = 0
            roi2 += emoj

        # cv2.putText(frame,'_',(640-j,145+h2*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)         #1 wall up
        # cv2.putText(frame,'_',(640-j-20,145+h2*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #2 wall up
        # cv2.putText(frame,'_',(640-j-40,145+h2*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #3 wall up

        # cv2.putText(frame,'_',(640-j,145+h2*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #1 wall down
        # cv2.putText(frame,'_',(640-j-20,145+h2*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)   #2 wall down
        # cv2.putText(frame,'_',(640-j-40,145+h2*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)   #3 wall down

        # cv2.line(frame,(640-j-40,0),(640-j-40,145+h2*10-1),(0, 0, 255),1)        #1 wall
        # cv2.line(frame,(640-j-40,480),(640-j-40,145+h2*10+70-3),(0, 0, 255),1)   #2 wall

        walltwo = []
        for count in range(0, 40):
            wallone.append((640-j-count, 145+h2*10-1))
            wallone.append((640-j-count, 145+h2*10+70-3))

        for item1 in range(145+h2*10-1):
            walltwo.append((640-j-40, item1))

        for item2 in range(145+h2*10+70-3, 480):
            walltwo.append((640-j-40, item2))

        if set(submarine_coordinate).intersection(walltwo) != set():
            # Crash(score)
            GameOverNotify = 999

    h = random.randint(0, 10)
    if pos > 320:
        k = j - 160
        if k == 0:
            h3 = h
            k = 1
        if k < 0:
            k = 640+k
        if k < 600:
            if k == 380:
                score = score + 1
            emoj = cv2.imread(f'{h3}.png')

            emoj = cv2.resize(emoj, (40, 478))
            img2gray = cv2.cvtColor(emoj, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            # cv2.putText(frame,f'.',(i, h1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
            roi3 = frame[-478-1:-1, -40-k:-k]

            roi3[np.where(mask)] = 0
            roi3 += emoj
        # cv2.putText(frame,'_',(640-k,145+h3*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)         #1 wall up
        # cv2.putText(frame,'_',(640-k-20,145+h3*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #2 wall up
        # cv2.putText(frame,'_',(640-k-40,145+h3*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #3 wall up

        # cv2.putText(frame,'_',(640-k,145+h3*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #1 wall down
        # cv2.putText(frame,'_',(640-k-20,145+h3*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)   #2 wall down
        # cv2.putText(frame,'_',(640-k-40,145+h3*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)   #3 wall down

        # cv2.line(frame,(640-k-40,0),(640-k-40,145+h3*10-1),(0, 0, 255),1)        #1 wall
        # cv2.line(frame,(640-k-40,480),(640-k-40,145+h3*10+70-3),(0, 0, 255),1)   #2 wall

        wallthree = []
        for count in range(0, 40):
            wallthree.append((640-k-count, 145+h3*10-1))
            wallthree.append((640-k-count, 145+h3*10+70-3))

        for item1 in range(145+h3*10-1):
            wallthree.append((640-k-40, item1))

        for item2 in range(145+h3*10+70-3, 480):
            wallthree.append((640-k-40, item2))

        if set(submarine_coordinate).intersection(wallthree) != set():
            GameOverNotify = 999

    h = random.randint(0, 10)
    if pos > 480:
        l = k - 160
        if l == 0:
            h4 = h
            l = 1
        if l < 0:
            l = 640+l
        if l < 600:
            if l == 380:
                score = score + 1
            emoj = cv2.imread(f'{h4}.png')

            emoj = cv2.resize(emoj, (40, 478))
            img2gray = cv2.cvtColor(emoj, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            # cv2.putText(frame,f'.',(i, h1),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
            roi4 = frame[-478-1:-1, -40-l:-l]

            roi4[np.where(mask)] = 0
            roi4 += emoj
        # cv2.putText(frame,'_',(640-l,145+h4*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)         #1 wall up
        # cv2.putText(frame,'_',(640-l-20,145+h4*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #2 wall up
        # cv2.putText(frame,'_',(640-l-40,145+h4*10-1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #3 wall up

        # cv2.putText(frame,'_',(640-l,145+h4*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)      #1 wall down
        # cv2.putText(frame,'_',(640-l-20,145+h4*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)   #2 wall down
        # cv2.putText(frame,'_',(640-l-40,145+h4*10+70-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1)   #3 wall down

        # cv2.line(frame,(640-l-40,0),(640-l-40,145+h4*10-1),(0, 0, 255),1)        #1 wall
        # cv2.line(frame,(640-l-40,480),(640-l-40,145+h4*10+70-3),(0, 0, 255),1)   #2 wall

        wallfour = []
        for count in range(0, 40):
            wallfour.append((640-l-count, 145+h4*10-1))
            wallfour.append((640-l-count, 145+h4*10+70-3))

        for item1 in range(145+h4*10-1):
            wallfour.append((640-l-40, item1))

        for item2 in range(145+h4*10+70-3, 480):
            wallfour.append((640-l-40, item2))

        if set(submarine_coordinate).intersection(wallfour) != set():
            GameOverNotify = 999
            

    cv2.rectangle(frame, (0, 0), (640, 10), (0, 0, 0), -1)  # top
    cv2.rectangle(frame, (0, 470), (640, 480), (0, 0, 0), -1)  # bottom
    cv2.rectangle(frame, (0, 0), (10, 480), (0, 0, 0), -1)  # left
    cv2.rectangle(frame, (630, 0), (640, 480), (0, 0, 0), -1)  # right

    cv2.rectangle(frame, (0, 0), (180, 40), (0, 0, 0), -1)  # score
    cv2.putText(frame, f'Score : {score}', (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


    if GameOverNotify == 999 :
        cv2.rectangle(frame, (200, 150), (440, 280), (0, 0, 0), -1)
        cv2.putText(frame, 'GAME OVER', (230, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # 1 wall up
        cv2.putText(frame, f'score : {score}', (235, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)  # 1 wall up

        GameOver = 1
        print('GameOver')

    pointer = pointer+speed
    pos = pos+speed

    LastPx = int(x1)
    LastPy = int(y1)

    cTime = time.time()
    # print(cTime)
    # print(pTime)
    fps = 1.0/float(cTime-pTime)
    pTime = cTime
    # frame = cv2.flip(frame, 1)
    cv2.imshow("image", frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break



cap.release()

cv2.destroyAllWindows()
