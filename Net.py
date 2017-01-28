import cv2
import numpy as np


kNearest = cv2.ml.KNearest_create()

def loadAndTrain():
    # read in training classifications
    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        os.system("pause")
        return False

    # read in training images
    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        os.system("pause")
        return False

    # reshape numpy array to 1d, necessary to pass to call to train
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    # set default K to 1
    kNearest.setDefaultK(1)

    # train KNN object
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    # network is initialized
    return True
