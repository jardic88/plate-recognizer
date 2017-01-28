import cv2
import numpy as np


class Plate:

    def __init__(self):
        self.text = ''
        self.plateCrop = None
        self.cropUpscaled = None
        self.frameThresh = None
        self.locationInFrame = None

