# __author__ = 'Subhodip Biswas'
import cv2
import numpy as np
from common import getIntersection
from centerDetect import locatedEyeCenter

# Global variables
mainWindowName = "Main Feed"
faceWindowName = "Face Feed"
leftEyeWindowName = "Left Eye"
rightEyeWindowName = "Right Eye"
constantSmoothFaceFactor = 0.05
constantSmoothFaceImage = False

cv2.namedWindow(mainWindowName, cv2.WINDOW_NORMAL)
# Face Window
cv2.namedWindow(faceWindowName, cv2.WINDOW_NORMAL)
# Left eye window
cv2.namedWindow(leftEyeWindowName, cv2.WINDOW_NORMAL)
# Right eye window
cv2.namedWindow(rightEyeWindowName, cv2.WINDOW_NORMAL)  # Use captured face and detect eye location
eyeTopPercent = 22
eyeSidePercent = 15
eyeHeightPercent = 30
eyeWidthPercent = 35


def locateEyes(frame, face):
    # Pythonic way to set region of interest.
    roiFace = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]:]
    debugFace = roiFace
    # Apply Gaussian Blur to prevent outliers from stuff like glasses
    cv2.blur(roiFace, (5, 5))  # TODO:Change or make the blur filter better
    # Draw eye region
    eyeWidth = face[2] * (eyeWidthPercent / 100.0)
    eyeHeight = face[2] * (eyeHeightPercent / 100.0)
    eyeTop = face[3] * (eyeTopPercent / 100.0)
    # Rectangle
    leftStartX = int(face[2] * (eyeSidePercent / 100.0))
    leftStartY = int(eyeTop)
    leftEndX = int(int(face[2] * (eyeSidePercent / 100.0)) + eyeWidth)
    leftEndY = int(int(eyeTop) + eyeHeight)
    rightStartX = int(face[2]) - (int(face[2] * (eyeSidePercent / 100.0)))
    rightStartY = int(eyeTop)
    rightEndX = int(face[2]) - (int(int(face[2] * (eyeSidePercent / 100.0)) + eyeWidth))
    rightEndY = int(int(eyeTop) + eyeHeight)
    # Line Coordinates
    leftStartHLineX = int(face[2] * (eyeSidePercent / 100.0))
    leftStartHLineY = (int(eyeTop) + int(eyeHeight / 2))
    leftEndHLineX = int(face[2] * (eyeSidePercent / 100.0)) + int(eyeWidth)
    leftEndHLineY = (int(eyeTop) + int(eyeHeight / 2))
    leftStartVLineX = int(face[2] * (eyeSidePercent / 100.0)) + int(eyeWidth / 2)
    leftStartVLineY = int(eyeTop)
    leftEndVLineX = int(face[2] * (eyeSidePercent / 100.0)) + int(eyeWidth / 2)
    leftEndVLineY = int(eyeTop) + int(eyeHeight)

    rightStartHLineX = int(face[2]) - int(face[2] * (eyeSidePercent / 100.0))
    rightStartHLineY = int(eyeTop) + int(eyeHeight / 2)
    rightEndHLineX = (int(face[2]) - int(face[2] * (eyeSidePercent / 100.0))) - int(eyeWidth)
    rightEndHLineY = int(eyeTop) + int(eyeHeight / 2)
    rightStartVLineX = int(face[2]) - int(face[2] * (eyeSidePercent / 100.0)) - int(eyeWidth / 2)
    rightStartVLineY = int(eyeTop)
    rightEndVLineX = int(face[2]) - int(face[2] * (eyeSidePercent / 100.0)) - int(eyeWidth / 2)
    rightEndVLineY = int(eyeTop) + int(eyeHeight)
    # Find the intersection between two lines and this is the center.
    # Left Center
    leftCenter = getIntersection([leftStartHLineX, leftStartHLineY, leftEndHLineX, leftEndHLineY],
                                 [leftStartVLineX, leftStartVLineY, leftEndVLineX, leftEndVLineY])
    # Right Center
    rightCenter = getIntersection([rightStartHLineX, rightStartHLineY, rightEndHLineX, rightEndHLineY],
                                  [rightStartVLineX, rightStartVLineY, rightEndVLineX, rightEndVLineY])

    # The center for both eye as coordinates and list to be send to socket.io
    #print [leftCenter.x, leftCenter.y, rightCenter.x, rightCenter.y]
    leftEyeRegion = roiFace[leftStartY: leftStartY + int(eyeHeight), leftStartX: leftStartX + int(eyeWidth):]
    rightEyeRegion = roiFace[rightStartY: rightStartY + int(eyeHeight), rightEndX: rightEndX + int(eyeWidth):]

    # Send the discovered region to find pupil and corner. #Main Code of this system
    leftEyePupil = locatedEyeCenter(roiFace, leftEyeRegion, leftEyeWindowName, eyeWidth)
    rightEyePupil = locatedEyeCenter(roiFace, rightEyeRegion, rightEyeWindowName, eyeWidth)

    # TODO: Currently side by side but give in between
    cv2.rectangle(roiFace, (leftStartX, leftStartY), (leftEndX, leftEndY),
                  (255, 255, 0))

    cv2.rectangle(roiFace, (rightStartX, rightStartY), (rightEndX, rightEndY),
                  (255, 255, 0))
    # Draw Lines in each of the above rectangles
    # left horizontal line
    cv2.line(roiFace, (leftStartHLineX, leftStartHLineY),
             (leftEndHLineX, leftEndHLineY), (255, 0, 0))
    # left vertical line
    cv2.line(roiFace, (leftStartVLineX, leftStartVLineY),
             (leftEndVLineX, leftEndVLineY), (255, 0, 0))
    # right horizontal line
    cv2.line(roiFace, (rightStartHLineX, rightStartHLineY),
             (rightEndHLineX, rightEndHLineY),
             (255, 0, 0))
    # right vertical line
    cv2.line(roiFace, (rightStartVLineX, rightStartVLineY),
             (rightEndVLineX, rightEndVLineY), (255, 0, 0))
    # Detect Eye
    # Mark Eye

    #print leftEyeRegion
    #left left
    leftRightEyeCornerRegion = leftEyeRegion
    leftPupilX, leftPupilY = leftEyePupil
    rightPupilX, rightPupilY = rightEyePupil
    leftRightEyeCornerRegion[2] -= leftPupilX
    leftRightEyeCornerRegion[0] += leftPupilX
    leftRightEyeCornerRegion[3] /= 2
    leftRightEyeCornerRegion[1] += leftRightEyeCornerRegion[3] / 2
    #left right
    leftLeftCornerRegion = leftEyeRegion
    leftLeftCornerRegion[2] -= leftPupilX
    leftLeftCornerRegion[3] /= 2
    leftLeftCornerRegion[1] += leftLeftCornerRegion[3] /2
    # right left
    rightLeftCornerRegion = rightEyeRegion
    rightLeftCornerRegion[2] -= rightPupilX
    rightLeftCornerRegion[3] /= 2
    rightLeftCornerRegion[1] += rightLeftCornerRegion[3]/ 2
    # right right
    rightRightCornerRegion = rightEyeRegion
    rightRightCornerRegion[2] -= rightPupilX
    rightRightCornerRegion[0] += rightPupilX
    rightRightCornerRegion[3] /= 2
    rightRightCornerRegion[1] += rightRightCornerRegion[3] /2

    #print leftRightEyeCornerRegion

    '''cv2.rectangle(debugFace, (leftRightEyeCornerRegion[0], leftRightEyeCornerRegion[1]),
                  (leftRightEyeCornerRegion[0]+leftRightEyeCornerRegion[2],
                   leftRightEyeCornerRegion[1]+leftRightEyeCornerRegion[3]),(0,255,0), 2)'''


    print rightEyePupil
    cv2.circle(debugFace, rightEyePupil, 3, 1234)
    cv2.circle(debugFace, leftEyePupil, 3, 1234)  # Calculate Deviation
    try:
        cv2.imshow(faceWindowName, roiFace)
    except e:
        print e  # Capture video feed from the webcam.
def videoFeedCapture():
    capturedFeed = cv2.VideoCapture(0)
    try:
        face_cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_alt.xml")
        eye_cascade = cv2.CascadeClassifier("../data/haarcascade_eye.xml")
    except:
        print "Error loading classifier trainers"

    while (True):
        ret, frame = capturedFeed.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        '''  This Section applies certain prerequisites to the video feed. In production there is no need for all the
        windows to be opened so suppress them'''
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            # cv2.imshow(rightEyeWindowName, gray)
        if type(faces).__module__ == np.__name__:
            print "face detected"
            locateEyes(gray, faces[0])
        cv2.imshow(mainWindowName, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capturedFeed.release()
    cv2.destroyAllWindows()
    # Mirror video feed frame for eye detect algos
    #  Mirror or use for left and right eye.
    # Expose as api


if __name__ == "__main__":
    videoFeedCapture()
