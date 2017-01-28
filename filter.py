import cv2
import numpy as np
import math


GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

def threshold(frame):
    frameGray = extractValue(frame)
    frameGrayMaxContrast = maximizeContrast(frameGray)
    height, width = frameGray.shape
    frameBlurred = np.zeros((height, width, 1), np.uint8)
    frameBlurred = cv2.GaussianBlur(frameGrayMaxContrast, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    frameThresh = cv2.adaptiveThreshold(frameBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    return frameThresh

def extractValue(frame):
    height, width, numChannels = frame.shape
    hsv = np.zeros((height, width, 3), np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return v

def maximizeContrast(frameGray):
    height, width = frameGray.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(frameGray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(frameGray, cv2.MORPH_BLACKHAT, structuringElement)

    frameGrayPlusTopHat = cv2.add(frameGray, imgTopHat)
    frameGrayPlusTopHatMinusBlackHat = cv2.subtract(frameGrayPlusTopHat, imgBlackHat)

    return frameGrayPlusTopHatMinusBlackHat
