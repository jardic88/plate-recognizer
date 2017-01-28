import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate
import Net
import color


showSteps = False


def main():

	# try to load and train neural network
	if Net.loadAndTrain() == False:
		print "\nerror: KNN traning was not successful"
		return

	imgOriginalScene = cv2.imread("../fotky/orez-velky.png")
	if imgOriginalScene is None:
		print "\nerror: image not read from file"
		os.system("pause")
		return

	listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
	listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

	cv2.imshow("imgOriginalScene", imgOriginalScene)

	if len(listOfPossiblePlates) == 0:
		print "\nno license plates were detected"
	else:
		# if we get in here list of possible plates has at leat one plate
		# sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
		listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

		# suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
		licPlate = listOfPossiblePlates[0]

		# show crop of plate and threshold of plate
		cv2.imshow("imgPlate", licPlate.imgPlate)
		cv2.imshow("imgThresh", licPlate.imgThresh)

		# if no chars were found in the plate
		if len(licPlate.strChars) == 0:
			print "\nno characters were detected\n\n"
			return

		# draw red rectangle around plate
		drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

		# write license plate text to std out
		print "\nlicense plate read from image = " + licPlate.strChars + "\n"
		print "----------------------------------------"

		# write license plate text on the image
		writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

		# re-show scene image
		cv2.imshow("imgOriginalScene", imgOriginalScene)


	# hold windows open until user presses a key
	cv2.waitKey()

	# remove windows from memory
	cv2.destroyAllWindows()

	return


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
	# get 4 vertices of rotated rect
	p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

	# draw 4 red lines
	cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), color.RED, 2)
	cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), color.RED, 2)
	cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), color.RED, 2)
	cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), color.RED, 2)


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
	# this will be the center of the area the text will be written to
	ptCenterOfTextAreaX = 0
	ptCenterOfTextAreaY = 0

	# this will be the bottom left of the area that the text will be written to
	ptLowerLeftTextOriginX = 0
	ptLowerLeftTextOriginY = 0

	sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
	plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

	# choose a plain jane font
	intFontFace = cv2.FONT_HERSHEY_SIMPLEX
	# base font scale on height of plate area
	fltFontScale = float(plateHeight) / 30.0
	# base font thickness on font scale
	intFontThickness = int(round(fltFontScale * 1.5))

	# call getTextSize
	textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)

	# unpack roatated rect into center point, width and height, and angle
	( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

	# make sure center is an integer
	intPlateCenterX = int(intPlateCenterX)
	intPlateCenterY = int(intPlateCenterY)

	# the horizontal location of the text area is the same as the plate
	ptCenterOfTextAreaX = int(intPlateCenterX)

	if intPlateCenterY < (sceneHeight * 0.75):
		# if the license plate is in the upper 3/4 of the image
		# write the chars in below the plate
		ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
	else:
		# else if the license plate is in the lower 1/4 of the image
		# write the chars in above the plate
		ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))

	# unpack text size width and height
	textSizeWidth, textSizeHeight = textSize

	# calculate the lower left origin of the text area
	ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
	# based on the text area center, width, and height
	ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

	# write the text on the image
	cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, color.YELLOW, intFontThickness)



if __name__ == "__main__":
	main()


















