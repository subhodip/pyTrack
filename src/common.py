from shapely.geometry import LineString
import numpy as np
import math
import cv2


def getIntersection(horizontal, vertical):
    horizontalLine = LineString([(horizontal[0], horizontal[1]), (horizontal[2], horizontal[3])])
    verticalLine = LineString([(vertical[0], vertical[1]), (vertical[2], vertical[3])])
    return horizontalLine.intersection(verticalLine)


def matrixMag(gradX, gradY):
    rows, cols = gradX.shape
    mags = np.zeros((rows, cols), dtype=np.float64)
    for y in range(rows):
        Xr = gradX[y]
        Yr = gradY[y]
        Mr = mags[y]
        for x in range(cols):
            gX = Xr[x]
            gY = Yr[x]
            magnitude = math.sqrt((gX ** 2) + (gY ** 2))
            Mr[x] = magnitude
    return mags


def computeDynamicThreshold(mags, thresholdFactor):
    rows, cols = mags.shape
    meanDevMag, stdDevMag = cv2.meanStdDev(mags)
    stdDev = stdDevMag[0][0] / math.sqrt(rows * cols)
    return thresholdFactor * stdDev + meanDevMag[0][0]
