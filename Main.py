import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import Net
import color


showSteps = False

# minimum number of characters the plate has to have
MIN_PLATE_CHARS = 6



def openVideo(path, sinceFrameNumber = 0, crop = 2.5):
	cap = cv2.VideoCapture(path)
	i = 0

	width = 0
	heightCropPoint = 0 
	height = 0

	while(cap.isOpened()):
		i = i + 1
		ret, frame = cap.read()
		if frame is None:
			break
		if i == 1 and crop > 0:
			height, width, x = frame.shape
			heightCropPoint = height - int(height / crop)
		# zacni az od 'sinceFrameNumber' casu
		if i < sinceFrameNumber:
			continue
		# vem jen kazdy druhy snimek
		if i % 2 == 0:
			# orezat frame
			if crop > 0:
				frame = frame[heightCropPoint : height, 0 : width]
			# zpracovat frame
			processFrame(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


def openImage(path):
	frame = cv2.imread(path)
	if frame is None:
		print "error: data not read from file"
		os.system("pause")
	else:
		processFrame(frame)


def processFrame(frame):

	possiblePlates = DetectPlates.detectPlatesInScene(frame)
	possiblePlates = DetectChars.detectCharsInPlates(possiblePlates)

	platesFound = len(possiblePlates)
	#print("plates: " + str(platesFound))
	for plate in possiblePlates:
		if len(plate.text) > MIN_PLATE_CHARS:
			print('PLATE: ' + plate.text)
			drawRedRectangleAroundPlate(frame, plate)
			writeLicensePlateCharsOnImage(frame, plate)
			#cv2.imshow('plateCrop', plate.plateCrop)
			#cv2.imshow('frameThresh', plate.frameThresh)
			#cv2.imwrite("../plateCrop.png", plate.plateCrop)

	cv2.imshow('frame', frame)


def main():

	# try to load and train neural network
	if Net.loadAndTrain() == False:
		print "error: KNN traning was not successful"
		return

	#openVideo('../fotky/1_0002.avi', 2.5, 170)
	#openVideo('../fotky/1_0002.avi', 170, 4)
	openVideo('../../../../../../1.avi', 900)
	#openVideo('../../../../../../1.avi', 1700)
	#openImage('../fotky/orez.png')

	# hold windows open until user presses a key
	cv2.waitKey()

	# remove windows from memory
	cv2.destroyAllWindows()

	return


def drawRedRectangleAroundPlate(frame, plate):
	# get 4 vertices of rotated rect
	rect = cv2.boxPoints(plate.locationInFrame)

	# draw 4 red lines
	cv2.line(frame, tuple(rect[0]), tuple(rect[1]), color.RED, 2)
	cv2.line(frame, tuple(rect[1]), tuple(rect[2]), color.RED, 2)
	cv2.line(frame, tuple(rect[2]), tuple(rect[3]), color.RED, 2)
	cv2.line(frame, tuple(rect[3]), tuple(rect[0]), color.RED, 2)


def writeLicensePlateCharsOnImage(frame, plate):
	# this will be the center of the area the text will be written to
	ptCenterOfTextAreaX = 0
	ptCenterOfTextAreaY = 0

	# this will be the bottom left of the area that the text will be written to
	ptLowerLeftTextOriginX = 0
	ptLowerLeftTextOriginY = 0

	sceneHeight, sceneWidth, sceneNumChannels = frame.shape
	plateHeight, plateWidth, plateNumChannels = plate.plateCrop.shape

	# choose a plain jane font
	intFontFace = cv2.FONT_HERSHEY_SIMPLEX
	# base font scale on height of plate area
	fltFontScale = float(plateHeight) / 30.0
	# base font thickness on font scale
	intFontThickness = int(round(fltFontScale * 1.5))

	# call getTextSize
	textSize, baseline = cv2.getTextSize(plate.text, intFontFace, fltFontScale, intFontThickness)

	# unpack roatated rect into center point, width and height, and angle
	( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = plate.locationInFrame

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
	cv2.putText(frame, plate.text, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, color.YELLOW, intFontThickness)



if __name__ == "__main__":
	main()


















