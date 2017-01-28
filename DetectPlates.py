import cv2
import numpy as np
import math
import Main
import random

import filter
import DetectChars
from Plate import Plate
from PlateChar import PlateChar
import color
from util import radiansToDegrees


# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

###################################################################################################
def detectPlatesInScene(imgOriginal):
	possiblePlates = []                   # this will be the return value

	height, width, numChannels = imgOriginal.shape

	frameThresh = np.zeros((height, width, 1), np.uint8)
	frameContours = np.zeros((height, width, 3), np.uint8)

	#cv2.destroyAllWindows()

	frameThresh = filter.threshold(imgOriginal)

	# find all possible chars in the scene,
	# this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
	possibleCharsInScene = findPossibleCharsInScene(frameThresh)

	# given a list of all possible chars, find groups of matching chars
	# in the next steps each group of matching chars will attempt to be recognized as a plate
	listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(possibleCharsInScene)

	# for each group of matching chars
	for matchingChars in listOfListsOfMatchingCharsInScene:
		# attempt to extract plate
		plate = extractPlate(imgOriginal, matchingChars)
		# if plate was found
		if plate.plateCrop is not None:
			# add to list of possible plates
			possiblePlates.append(plate)
	return possiblePlates


def findPossibleCharsInScene(frameThresh):
	possibleChars = []
	frameThreshCopy = frameThresh.copy()
	frameContours, contours, npaHierarchy = cv2.findContours(frameThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours
	height, width = frameThresh.shape
	frameContours = np.zeros((height, width, 3), np.uint8)
	# transform contours into chars and fill list with them
	for contour in contours:
		char = PlateChar(contour)
		# if contour is a possible char, note this does not compare to other chars (yet) . . .
		if char.isValid():
			# and add to list of possible chars
			possibleChars.append(char)
	return possibleChars


def extractPlate(frame, matchingChars):
	plate = Plate()

	# sort chars from left to right based on x position
	matchingChars.sort(key = lambda char: char.centerX)

	# calculate the center point of the plate
	plateCenterX = (matchingChars[0].centerX + matchingChars[len(matchingChars) - 1].centerX) / 2.0
	plateCenterY = (matchingChars[0].centerY + matchingChars[len(matchingChars) - 1].centerY) / 2.0
	plateCenter = plateCenterX, plateCenterY

	# calculate plate width and height
	plateWidth = int((matchingChars[len(matchingChars) - 1].x + matchingChars[len(matchingChars) - 1].width - matchingChars[0].x) * PLATE_WIDTH_PADDING_FACTOR)

	intTotalOfCharHeights = 0
	for char in matchingChars:
		intTotalOfCharHeights = intTotalOfCharHeights + char.height

	averageCharHeight = intTotalOfCharHeights / len(matchingChars)
	plateHeight = int(averageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

	# calculate correction angle of plate region
	opposite = matchingChars[len(matchingChars) - 1].centerY - matchingChars[0].centerY
	hypotenuse = DetectChars.getDistanceBetweenChars(matchingChars[0], matchingChars[len(matchingChars) - 1])
	radians = math.asin(opposite / hypotenuse)
	degrees = radiansToDegrees(radians)

	# pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
	plate.locationInFrame = (plateCenter, (plateWidth, plateHeight), degrees)
	# final steps are to perform the actual rotation
	# get the rotation matrix for our calculated correction angle
	rotationMatrix = cv2.getRotationMatrix2D(plateCenter, degrees, 1.0)
	# unpack original image width and height
	height, width, numChannels = frame.shape
	# rotate the entire image
	imgRotated = cv2.warpAffine(frame, rotationMatrix, (width, height))
	imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), plateCenter)
	# copy the cropped plate image into the applicable member variable of the possible plate
	plate.plateCrop = imgCropped

	return plate



