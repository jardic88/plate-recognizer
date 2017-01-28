import cv2
import numpy as np
import math


MIN_WIDTH = 2
MIN_HEIGHT = 8
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0
MIN_PIXEL_AREA = 80

class PlateChar:

	def __init__(self, contour):
		self.text = ''
		[self.x, self.y, self.width, self.height] = cv2.boundingRect(contour)
		self.intBoundingRectArea = self.width * self.height
		self.centerX = (self.x + self.x + self.width) / 2
		self.centerY = (self.y + self.y + self.height) / 2
		self.diagonal = math.sqrt((self.width ** 2) + (self.height ** 2))

	# this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
	# note that we are not (yet) comparing the char to other chars to look for a group
	def isValid(self):
		aspectRatio = float(self.width) / float(self.height)
		if (self.intBoundingRectArea > MIN_PIXEL_AREA and
			self.width > MIN_WIDTH and self.height > MIN_HEIGHT and
			MIN_ASPECT_RATIO < aspectRatio < MAX_ASPECT_RATIO):
			return True
		else:
			return False



