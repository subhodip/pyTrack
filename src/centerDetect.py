# __author__ = 'Subhodip Biswas'
import cv2
import numpy as np
import math
from shapely.geometry import Point
from common import matrixMag
from common import computeDynamicThreshold
import Queue

kFastEyeWidth = 50
kGradientThreshold = 50.0
kWeightBlurSize = 5
kEnableWeight = True
kWeightDivisor = 1.0
kEnablePostProcess = True
kPostProcessThreshold = 0.97
kPlotVectorField = False


def unScalePoint(point, eyeWidth):
    ratio = float(kFastEyeWidth) / eyeWidth
    pointX, pointY = point
    print ratio
    x = round(pointX / ratio, 2)
    y = round(pointY / ratio, 2)
    return (int(y), int(x))


# Imitating Matlabs gradient algorithm : thume
# Has overflow errors
def computeXGradient(imgMat):
    rows, cols = imgMat.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    for y in range(rows):
        Mr = imgMat[y]
        Or = out[y]
        Or[0] = Mr[1] - Mr[0]
        for x in range(cols - 1):
            Or[x] = (Mr[x + 1] - Mr[x - 1] / 2.0)
        Or[cols - 1] = Mr[cols - 1] - Mr[cols - 2]
    return out


def findPossibleCenter(x, y, weight, gX, gY, outSum):
    oRows, oCols = outSum.shape
    # for all centers
    for cy in range(oRows):
        Or = outSum[cy]
        Wr = weight[cy]
        for cx in range(oCols):
            if x == cx and y == cy:
                continue
            dx = x - cx
            dy = y - cy
        magnitude = math.sqrt(dx ** 2 + dy ** 2)
        dx = dx / magnitude
        dy = dy / magnitude
        dotProd = max(0.0, float(dx * gX + dx * gY))
        if kEnableWeight:
            Or[cx] += dotProd ** 2 * (Wr[cx] / kWeightDivisor)
        else:
            Or[cx] += dotProd ** 2

            # return outSum


def floodKillEdges(floodClone):
    fcRows, fcCols = floodClone.shape
    mask = np.uint8(floodClone)
    toDo = Queue.Queue()
    toDo.put((0, 0))
    if not toDo.empty():
        px, py = toDo.get()
        newp = (px + 1, py)
        toDo.put(newp)
        newp = (px - 1, py)
        toDo.put(newp)
        newp = (px, py + 1)
        toDo.put(newp)
        newp = (px, py - 1)

    return mask


def locatedEyeCenter(face, region, windowName, eyeWidth):
    # Fabian Timm's algorithmic way to detect gaze, not production ready but works
    # Help from thume C++ implementation
    roiEyeUnscaled = region
    # scaleToFastSize Function Start
    # srcRows, srcCols = roiEyeUnscaled.shape
    # try:
    #    roiEye = cv2.resize(roiEyeUnscaled, (kFastEyeWidth, int(float(kFastEyeWidth)/srcCols)*srcRows))
    #    print roiEye
    # except e:
    #    print e
    # Hackish way to get a 50% reduction in image size
    roiEye = cv2.resize(roiEyeUnscaled, (0, 0), fx=float(kFastEyeWidth) / 100, fy=float(kFastEyeWidth) / 100)
    eyeRows, eyeCols = roiEye.shape
    # scaleToFastSize Function End
    # Draw eye region
    # See in Main.py
    # Find Gradients
    gradientX = computeXGradient(roiEye)
    gradientY = computeXGradient(roiEye.transpose()).transpose()
    # compute all the magnitude
    magnitude = matrixMag(gradientX, gradientY)
    # compute the dynamic threshold
    gradThreshold = computeDynamicThreshold(magnitude, kGradientThreshold)
    # Normalize
    # Normalize and Threshold the gradients
    for y in range(eyeRows):
        Xr = gradientY[y]
        Yr = gradientY[y]
        Mr = magnitude[y]
        for x in range(eyeCols):
            gX = Xr[x]
            gY = Yr[x]
            mag = Mr[x]
            if (mag > gradThreshold):
                Xr[x] = gX / mag
                Yr[x] = gY / mag
            else:
                Xr[x] = 0.0
                Yr[x] = 0.0
    cv2.imshow(windowName, gradientX)
    # Blurred and Inverted image for weighting
    weight = cv2.GaussianBlur(roiEye, (kWeightBlurSize, kWeightBlurSize), 0)
    wRows, wCols = weight.shape
    for y in range(wRows):
        row = weight[y]
        for x in range(wCols):
            row[x] = (255 - row[x])
    # Algorithm - Reverse Loop
    outSum = np.zeros((eyeRows, eyeCols), dtype=np.float64)
    oRows, oCols = outSum.shape
    print "Eye Size: %s %s", oRows, oCols
    # Test for every center of each gradient
    for y in range(wRows):
        Xr = gradientX[y]
        Yr = gradientY[y]
        for x in range(wCols):
            gX = Xr[x]
            gY = Yr[x]
            if gX == 0.0 and gY == 0.0:
                continue
            findPossibleCenter(x, y, weight, gX, gY, outSum)
            # Scale Down values to average them

    numGrad = wRows * wCols
    out = np.float32(outSum)
    out = out * (1.0 / numGrad)
    maxMinArray = cv2.minMaxLoc(out)
    maxVal = maxMinArray[1]
    maxPoint = maxMinArray[3]
    if kEnablePostProcess:
        floodThreshold = maxVal * kPostProcessThreshold
        th, floodClone = cv2.threshold(out, floodThreshold, 0.0, cv2.THRESH_TOZERO)
        if kPlotVectorField:
            cv2.imwrite("eyeframe.png", roiEyeUnscaled)

        mask = floodKillEdges(floodClone)
        cv2.minMaxLoc(out, mask=mask)

    print unScalePoint(maxPoint, eyeWidth)
    return unScalePoint(maxPoint, eyeWidth)
