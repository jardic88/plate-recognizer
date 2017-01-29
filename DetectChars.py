# DetectChars.py

import cv2
import numpy as np
import math
import random

import Main
import filter
from PlateChar import PlateChar
import color
from Net import kNearest
from util import radiansToDegrees



# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_DIFF_IN_AREA = 0.5
MAX_DIFF_IN_WIDTH = 0.8
MAX_DIFF_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# other constants
MIN_MATCHING_CHARS = 3

TARGET_CHAR_WIDTH = 20
TARGET_CHAR_HEIGHT = 30


def detectCharsInPlates(possiblePlates):
	if len(possiblePlates) == 0:          # if list of possible plates is empty
		return possiblePlates             # return

	#intPlateCounter = 0
	#imgContours = None
	contours = []

	# at this point we can be sure the list of possible plates has at least one plate

	for plate in possiblePlates:
		plate.plateUpscaled = cv2.resize(plate.plateCrop, (0, 0), fx = 2, fy = 2)
		plate.plateThresh = filter.threshold(plate.plateUpscaled)

		# increase size of plate image for easier viewing and char detection
		#plate.plateThresh = cv2.resize(plate.plateThresh, (0, 0), fx = 3, fy = 3)
		# threshold again to eliminate any gray areas
		#thresholdValue, plate.plateThresh = cv2.threshold(plate.plateThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		# find all possible chars in the plate,
		# this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
		possibleCharsInPlate = findPossibleCharsInPlate(plate)

		# given a list of all possible chars, find groups of matching chars within the plate
		matchingCharsInPlate = findListOfListsOfMatchingChars(possibleCharsInPlate)
		matchingCharsInPlateLength = len(matchingCharsInPlate)

		# if no groups of matching chars were found in the plate
		if (matchingCharsInPlateLength == 0):
			plate.text = ""
			possiblePlates.remove(plate)
			continue

		# within each list of matching chars
		for i in range(0, matchingCharsInPlateLength):
			# sort chars from left to right
			matchingCharsInPlate[i].sort(key = lambda char: char.centerX)
			# and remove inner overlapping chars
			matchingCharsInPlate[i] = filterOverlappingChars(matchingCharsInPlate[i])

		# within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
		lengthOfLongestList = 0
		indexOfLongestList = 0

		# loop through all the vectors of matching chars, get the index of the one with the most chars
		for i in range(0, matchingCharsInPlateLength):
			if len(matchingCharsInPlate[i]) > lengthOfLongestList:
				lengthOfLongestList = len(matchingCharsInPlate[i])
				indexOfLongestList = i

		# suppose that the longest list of matching chars within the plate is the actual list of chars
		longestMatchingCharsInPlate = matchingCharsInPlate[indexOfLongestList]
		plate.text = recognizeTextInPlate(plate, longestMatchingCharsInPlate)

	return possiblePlates



def findPossibleCharsInPlate(plate):
	plateThresh = plate.plateThresh
	possibleChars = []
	#contours = []
	plateThreshCopy = plateThresh.copy()
	# find all contours in plate

	height, width = plateThresh.shape
	imgContours, contours, npaHierarchy = cv2.findContours(plateThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		char = PlateChar(contour)
		# if contour is a possible char, note this does not compare to other chars (yet) . . .
		if char.isValid():
			# add to list of possible chars
			possibleChars.append(char)
	return possibleChars



def findListOfListsOfMatchingChars(possibleChars):
	# with this function, we start off with all the possible chars in one big list
	# the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
	# note that chars that are not found to be in a group of matches do not need to be considered further
	listOfListsOfMatchingChars = []

	# for each possible char in the one big list of chars
	# find all chars in the big list that match the current char
	for char in possibleChars:
		matchingChars = findMatchingChars(char, possibleChars)
		# also add the current char to current possible list of matching chars
		matchingChars.append(char)
		# if current possible list of matching chars is not long enough to constitute a possible plate
		if len(matchingChars) < MIN_MATCHING_CHARS:
			continue
			# jump back to the top of the for loop and try again with next char, note that it's not necessary
			# to save the list in any way since it did not have enough chars to be a possible plate

		# if we get here, the current list passed test as a "group" or "cluster" of matching chars
		listOfListsOfMatchingChars.append(matchingChars)      # so add to our list of lists of matching chars
		# remove the current list of matching chars from the big list so we don't use those same chars twice,
		# make sure to make a new big list for this since we don't want to change the original big list
		possibleCharsWithCurrentMatchesRemoved = list(set(possibleChars) - set(matchingChars))
		recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(possibleCharsWithCurrentMatchesRemoved)      # recursive call
		# for each list of matching chars found by recursive call
		# add to our original list of lists of matching chars
		for recursiveMatchingChars in recursiveListOfListsOfMatchingChars:
			listOfListsOfMatchingChars.append(recursiveMatchingChars)
		break
	return listOfListsOfMatchingChars


def findMatchingChars(char, possibleChars):
	# the purpose of this function is, given a possible char and a big list of possible chars,
	# find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
	matchingChars = []                # this will be the return value

	# for each char in big list
	for possibleMatch in possibleChars:
		if possibleMatch == char:
			# if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
			# then we should not include it in the list of matches b/c that would end up double including the current char
			# so do not add to list of matches and jump back to top of for loop
			continue
		
		# compute stuff to see if chars are a match
		distanceBetweenChars = getDistanceBetweenChars(char, possibleMatch)
		angleBetweenChars = getAngleBetweenChars(char, possibleMatch)
		differenceInArea = float(abs(possibleMatch.intBoundingRectArea - char.intBoundingRectArea)) / float(char.intBoundingRectArea)
		differenceInWidth = float(abs(possibleMatch.width - char.width)) / float(char.width)
		differenceInHeight = float(abs(possibleMatch.height - char.height)) / float(char.height)

		# check if chars match
		if (distanceBetweenChars < (char.diagonal * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
			angleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
			differenceInArea < MAX_DIFF_IN_AREA and
			differenceInWidth < MAX_DIFF_IN_WIDTH and
			differenceInHeight < MAX_DIFF_IN_HEIGHT):
			# if the chars are a match, add the current char to list of matching chars
			matchingChars.append(possibleMatch)

	return matchingChars


def getAbsoluteCenter(firstChar, secondChar):
	x = abs(firstChar.centerX - secondChar.centerX)
	y = abs(firstChar.centerY - secondChar.centerY)
	return x, y


# use Pythagorean theorem to calculate distance between two chars
def getDistanceBetweenChars(firstChar, secondChar):
	x, y = getAbsoluteCenter(firstChar, secondChar)
	return math.sqrt((x ** 2) + (y ** 2))


# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def getAngleBetweenChars(firstChar, secondChar):
	x, y = getAbsoluteCenter(firstChar, secondChar)
	adjacent = float(x)
	opposite = float(y)

	# check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
	if adjacent != 0.0:
		# if adjacent is not zero, calculate angle
		radians = math.atan(opposite / adjacent)
	else:
		# if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
		radians = 1.5708

	# angle in degrees
	return radiansToDegrees(radians)


# if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
# this is to prevent including the same char twice if two contours are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should only include the char once
def filterOverlappingChars(matchingChars):
	filteredChars = list(matchingChars)

	for char in matchingChars:
		for otherChar in matchingChars:
			# if current char and other char are not the same char . . .
			if char != otherChar:
				# if current char and other char have center points at almost the same location . . .
				if getDistanceBetweenChars(char, otherChar) < (char.diagonal * MIN_DIAG_SIZE_MULTIPLE_AWAY):
					# if we get in here we have found overlapping chars
					# next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
					if char.intBoundingRectArea < otherChar.intBoundingRectArea:         # if current char is smaller than other char
						if char in filteredChars:              # if current char was not already removed on a previous pass . . .
							filteredChars.remove(char)         # then remove current char
					else:                                                                       # else if other char is smaller than current char
						if otherChar in filteredChars:                # if other char was not already removed on a previous pass . . .
							filteredChars.remove(otherChar)           # then remove other char

	return filteredChars


# this is where we apply the actual char recognition
def recognizeTextInPlate(plate, matchingChars):
	plateThresh = plate.plateThresh
	plateUpscaled = plate.plateUpscaled
	text = ""
	#i = 0
	# for each char in plate
	for char in matchingChars:
		#i = i + 1
		# crop char out of threshold image
		x = char.x
		y = char.y
		charCrop = plateThresh[y : y + char.height, x : x + char.width]
		#charCrop2 = plateUpscaled[y : y + char.height, x : x + char.width]
		#cv2.imshow('plateUpscaled' + str(i), charCrop2)

		# resize image, this is necessary for char recognition
		charCrop = cv2.resize(charCrop, (TARGET_CHAR_WIDTH, TARGET_CHAR_HEIGHT))
		# flatten image into 1d numpy array
		charCrop = charCrop.reshape((1, TARGET_CHAR_WIDTH * TARGET_CHAR_HEIGHT))
		# convert from 1d numpy array of ints to 1d numpy array of floats
		charCrop = np.float32(charCrop)

		retval, npaResults, neigh_resp, dists = kNearest.findNearest(charCrop, k = 1)
		char.text = chr(int(npaResults[0][0]))
		text = text + char.text

	return text

