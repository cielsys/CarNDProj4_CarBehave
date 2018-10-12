import os
import errno
import csv
import random
import math
import datetime
import pickle
import threading

import cv2
import numpy as np

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#====================== CTrainingRecord() =====================
class CTrainingRecord():
    '''
    Convenience class for holding all of the neccesary info
    for a single image.
    '''
    # ----------------------- ctor
    def __init__(self, fileName, steeringAngle, whichCam, doMirror):
        self.fileName = fileName
        self.steeringAngle = steeringAngle
        self.whichCam = whichCam
        self.doMirror = doMirror

    # --------------------------------- GetImage()
    def GetImage(self):
        '''
        Read the corresponding image into standard numpy
        HxWxRBG array. Mirror the image if specified.
        :return:
        '''
        img = mpimg.imread(self.fileName)
        if self.doMirror:
            img = np.fliplr(img)
        return(img)

    # --------------------------------- GetImageShape()
    def GetImageShape(self):
        img = mpimg.imread(self.fileName)
        shape = img.shape()
        return(shape)

    # --------------------------------- PlotImage
    def PlotImage(self, figtitle=None):
        if figtitle is None:
            figtitle = "{}".format(self.fileName)

        img = self.GetImage()
        axTitle = "cam={} mir={} steer={} shape={}".format(self.whichCam, self.doMirror, self.steeringAngle, img.shape)
        PlotImage(img, figtitle, axTitle)

# --------------------------------- PlotImage
def PlotImage(img, figtitle, axTitle):
    figsize = (8, 4)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(figtitle, fontsize=10)

    ax = plt.gca()
    ax.imshow(img, interpolation='sinc')
    ax.set_title(axTitle, fontsize=8)
    plt.tight_layout(pad=2)
    plt.show()

#--------------------------------- SimpleMultiImage
def SimpleMultiImage(imgInList, figtitle="TestImg"):
    figsize = (9, 3)

    plotNumCols= len(imgInList)
    plotNumRows = 1

    fig, axes = plt.subplots(plotNumRows, plotNumCols, figsize=figsize, )#subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle(figtitle, fontsize=16)

    for (imageIndex, (ax, imgOut)) in enumerate(zip(axes.flat, imgInList)):
        #imgOut = imgInList[imageIndex]
        title = "img[{}]".format(imageIndex)
        #ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.imshow(imgOut, interpolation='sinc') #dsIn.X[imageIndex]

    plt.tight_layout() # (pad=-0.75 w_pad=0.5, h_pad=1.0)
    plt.show()

#--------------------------------- ReadCSVFile()
def CSVRowRawToFields(csvRowRaw):
    (fnameCenter, fnameLeft, fnameRight, steeringStr, throttleStr, brakeStr, speedStr) = csvRowRaw
    return (fnameCenter, fnameLeft, fnameRight, steeringStr, throttleStr, brakeStr, speedStr)

#--------------------------------- ReadCSVFile()
def ReadCSVFile(csvFileNameIn, limitSize=0):
    '''
    Read each row of the image meta data CSV and fixup the filenames.
    This is because the filenames, in older versions of the recorder,
    included full path info that was system dependent.
    This function assumes that the file basesnames can be found
    in a directory named "IMG" residing in the same folder as
    the CSV file.
    :param csvFileNameIn:
    :param limitSize: Use 0 for normal operation. Positive values are used for dev/debug only
    :return: A list of filepath fixed CSV row records
    '''
    print("Reading CSV file: '{}'... ".format(csvFileNameIn), end='', flush=True)
    csvFilePath = os.path.dirname(csvFileNameIn)
    imageDir = csvFilePath + "/IMG/"

    csvRowsRaw = []
    numRows = 0
    with open(csvFileNameIn, mode='r') as infile:
        infile.readline()  # Skip header line 0
        reader = csv.reader(infile)

        for csvRowRaw in csv.reader(infile):
            (fnameCenter, fnameLeft, fnameRight, steeringStr, throttleStr, brakeStr, speedStr) = CSVRowRawToFields(csvRowRaw)

            csvRowRaw[0] = imageDir + os.path.basename(fnameCenter)
            csvRowRaw[1] = imageDir + os.path.basename(fnameLeft)
            csvRowRaw[2] = imageDir + os.path.basename(fnameRight)
            csvRowsRaw.append(csvRowRaw)

            numRows+=1
            if (limitSize > 0) and (limitSize == numRows):
                print("LIMITING INPUT to {} rows. ".format(limitSize), end='')
                break

    print("Done reading {} rows".format(numRows))
    return (csvRowsRaw)

#--------------------------------- CSVRawRowsToTrainingRecs()
def CSVRawRowsToTrainingRecs(csvRowsRaw, camAngleCorrection):
    '''
    Read each CSV row record and demultiplex/create
    an independent CTrainingRecord for each of the 3 camera images.
    Side camera images adjust the steering angle by camAngleCorrection
    :param csvRowsRaw:
    :param camAngleCorrection:
    :return: 2 Lists of CTrainingRecords, one of Center cam and one of Side cams
    '''
    trainRecsCenter = []
    trainRecsSides = []

    for csvRowRaw in csvRowsRaw:
        (fnameCenter, fnameLeft, fnameRight, steeringStr, throttleStr, brakeStr, speedStr) = CSVRowRawToFields(csvRowRaw)
        steeringVal = float(steeringStr)
        trainRecsCenter.append(CTrainingRecord(fnameCenter, steeringVal, whichCam="center", doMirror=False))
        trainRecsSides.append(CTrainingRecord(fnameLeft, steeringVal + camAngleCorrection, whichCam="left", doMirror=False))
        trainRecsSides.append(CTrainingRecord(fnameRight, steeringVal - camAngleCorrection, whichCam="right", doMirror=False))

    return(trainRecsSides, trainRecsCenter)

#--------------------------------- RecordToString()
def RecordToString(rec):
    '''
    Pretty print formatting of a single training Record.
    :param rec:
    :return:
    '''
    recStr = "fn={:>75}, steer= {:+0.2f}, cam= {:>6}, doMirror= {}".format(rec.fileName,rec.steeringAngle, rec.whichCam, rec.doMirror)
    return(recStr)

#--------------------------------- PrintRecords()
def PrintRecords(trainRecs, numShow=0):
    '''
    Dev debug utility for inspecting lists of trainRecs.
    :param trainRecs:
    :param numShow: Number of recs to show, or 0 for all
    '''
    numShow = len(trainRecs) if (numShow <= 0 or numShow > len(trainRecs)) else numShow
    print("Showing {} of {} trainingRecords".format(numShow, len(trainRecs)))
    for recIndex, rec in enumerate(trainRecs):
        recStr = RecordToString(rec)
        print("recs[{:6}]: {}".format(recIndex, recStr))
        if (numShow == recIndex+1):
            break

    print()


#--------------------------------- GeneratorThreadWrapper()
def GeneratorThreadWrapper(gen):
    '''
    Never used. This was to allow model.fit_generator() multi workers
    to improve CPU utilization during training.
    :param gen:
    :return:
    '''
    lock = threading.Lock()
    while True:
        try:
            with lock:
                x, y = next(gen)
        except StopIteration:
            return
        yield x, y

#--------------------------------- TrainRecordBatchGenerator()
# This code is for experimental HSV conversion
    #doConvertToHSV = False
    #if doConvertToHSV:
    #    imgCur = matplotlib.colors.rgb_to_hsv(imgCur)

def TrainRecordBatchGenerator(trainRecs, batchSize, cropTBLR=None):
    '''
    This generator supplies batches of X,y training images
    to keras.model.fit_generator() in model.py::Main()
    :param trainRecs: A list of training records to extract X,y batches from
    :param batchSize: Number of X,y values to provide per next()
    :param cropTBLR: X Image cropping spec
    :return: 2 batchSize Lists of X images, y steering angles
    '''
    numRecs = len(trainRecs)

    while True: # Never runs out - recycles the training recs if needed
        np.random.shuffle(trainRecs)

        for offset in range(0, numRecs, batchSize):
            trainRecsBatch = trainRecs[offset : offset + batchSize]

            XBatch = []
            yBatch = []
            for batchItemIndex, trainRecCur in enumerate(trainRecsBatch):
                imgCur = trainRecCur.GetImage() # Takes care of mirror internally, if needed

                # Crop the original image to Region of Interest
                if cropTBLR is not None:
                    imgCur = imgCur[cropTBLR[0]: - cropTBLR[1], :, :]

                XBatch.append(imgCur)
                yBatch.append(trainRecCur.steeringAngle)

            XBatch = np.array(XBatch)
            yBatch = np.array(yBatch)
            yield (XBatch, yBatch)

#====================== Main() =====================
def DevDebugMain(csvFileNameIn):
    '''
    This Main() is for dev debug only. This file is not normally called directly.
    :param csvFileNameIn:
    :return:
    '''

    #dir = "/home/cl/AAAProjects/AAAUdacity/carND/Proj4_CarBehave/Proj4Root/Assets/writeupImages/"
    #left=mpimg.imread(dir + "rawLeft.png")
    #center=mpimg.imread(dir + "rawCenter.png")
    #right=mpimg.imread(dir + "rawRight.png")
    #SimpleMultiImage([left, center, right], figtitle="Raw Left, Center, Right")

    limitSize = 0
    camAngleCorrection = 0.2
    csvRowsRaw = ReadCSVFile(csvFileNameIn, limitSize=limitSize)
    csvRowsRaw = csvRowsRaw[7833:7834]
    trainRecsSides, trainRecsCenter = CSVRawRowsToTrainingRecs(csvRowsRaw, camAngleCorrection=camAngleCorrection)
    PrintRecords(trainRecsSides, numShow=10)

    trainRecsCenterMirror = [CTrainingRecord(rec.fileName, -rec.steeringAngle, rec.whichCam, not rec.doMirror) for rec in trainRecsCenter]
    trainRecsFull = trainRecsCenter + trainRecsCenterMirror
    #PrintRecords(trainRecsFull, numShow=10)

    #np.random.shuffle(trainRecsFull)
    PrintRecords(trainRecsFull, numShow=10)

    #trainRecsSides[0].PlotImage()
    #trainRecsCenter[0].PlotImage()
    #trainRecsSides[1].PlotImage()

    imgCenterNoMirror = trainRecsFull[0].GetImage()
    imgCenterMirror = trainRecsFull[1].GetImage()

    cropTBLR = (60, 10, 0, 0)
    imgCenterNoMirror = imgCenterNoMirror[cropTBLR[0]: - cropTBLR[1], :, :]
    imgCenterMirror = imgCenterMirror[cropTBLR[0]: - cropTBLR[1], :, :]
    SimpleMultiImage([imgCenterNoMirror, imgCenterMirror], figtitle="Center,  NoMirror and Mirror. Cropped")


#====================== Main Invocation =====================
if ((__name__ == '__main__')):
    trainingFilesDirIn = "./Assets/trainingdata/"
    csvFileNameIn = trainingFilesDirIn + "driving_log.csv"

    DevDebugMain(csvFileNameIn)