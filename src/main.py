# __author__ = 'Subhodip Biswas'
import cv2

# Capture video feed from the webcam.
def videoFeedCapture():
    capturedFeed = cv2.VideoCapture(0)
    face_cascade = ""
    eye_cascade = ""
    mainWindowName = "Main window feed"
    faceWindowName = "Face window feed"
    leftEyeWindowName = "Left eye"
    rightEyeWindowName = "Right eye"
    while (True):
        ret, frame = capturedFeed.read()
        cv2.imshow(mainWindowName, frame)
        cv2.imshow(faceWindowName, frame)
        cv2.imshow(leftEyeWindowName, frame)
        cv2.imshow(rightEyeWindowName, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capturedFeed.release()
    cv2.destroyAllWindows()  # Mirror video feed frame for eye detect algos

# Mirror or use for left and right eye.
# Expose as api
if __name__ == "__main__":
    videoFeedCapture()
