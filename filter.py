import cv2
import numpy as np
import math
import random




THRESH_BLOCK_SIZE = 19
THRESH_WEIGHT = 9

# Otsu's Thresholding
# http://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html
def threshold(frame):
	frameGray = gray(frame)
	frameContrast = contrast(frameGray)
	frameBlurred = cv2.GaussianBlur(frameContrast, (5, 5), 0)
	frameThresh = cv2.adaptiveThreshold(frameBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, THRESH_BLOCK_SIZE, THRESH_WEIGHT)
	#cv2.imshow('frameThresh' + str(random.random()), frameThresh)
	return frameThresh

def gray(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	return v

def contrast(frame):
	structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	imgTopHat   = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, structuringElement)
	imgBlackHat = cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, structuringElement)
	return cv2.subtract(cv2.add(frame, imgTopHat), imgBlackHat)
