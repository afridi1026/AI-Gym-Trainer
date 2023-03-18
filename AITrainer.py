import cv2
import numpy as np
import time
import posemodule as pm

vid = cv2.VideoCapture(0)       # Reading video from webcam

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0
StraightHandAngle = 70
FlexedHandAngle = 160

while True:
    #img = cv2.imread("testimg.jpg")
    # img = cv2.flip(img,1)

    flag, img = vid.read()
    img = cv2.resize(img, (1080, 720))

    # Calling findpose method from poseDetector class to draw landmarks of body
    detector.findPose(img, False)

    # Function to get the coordinates on image of all landmarks
    # It returns a list of of lists with id and x,y values of all landmarks
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # Getting the angle between or desired three points among landmarks
        angle = detector.findangle(img, 12, 14, 16)

        # Changing the range of our angles between 0 to 100
        per = np.interp(angle, (StraightHandAngle, FlexedHandAngle), (100,0))

        # Creating a variable with approximately equal range to angle
        bar = np.interp(angle, (80,160), (100,650))
        #print(angle, per)

        # Check for curls
        color = (255,0,255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1

        if per == 0:
            color = (0, 0, 255)
            if dir == 1:
                count += 0.5
                dir = 0

        # DRAW BAR

        cv2.rectangle(img, (980, 100), (1055, 650), color, 3)
        cv2.rectangle(img, (980, int(bar)), (1055, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (940, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)


        # DRAW CURL COUNT

        cv2.rectangle(img, (0,520),(200,720),(255,255,255),cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 680), cv2.FONT_HERSHEY_PLAIN, 10, (255,0, 0), 20)
        #print(count)
        #cv2.putText(img, f'Curls Done : {int(count)}', (50,100), cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

    # Draw FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS : {int(fps)}', (25, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if (k == ord(' ')):
        break

cv2.destroyAllWindows()
vid.release()