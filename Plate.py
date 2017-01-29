import cv2
import numpy as np


class Plate:

    def __init__(self):
        self.text = ''
        self.plateCrop = None
        self.plateUpscaled = None
        self.plateThresh = None
        self.locationInFrame = None

