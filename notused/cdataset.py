import os
import glob
import pickle
import csv
import re
import time
from collections import OrderedDict
import copy

import numpy as np
import matplotlib.image as mpimg

g_ConvertImagesToTensors = False

#====================== CDataSet() =====================
class CDataSet():
    # ----------------------- ctor
    def __init__(self, name="NoName", pickleFileNameIn = None, dictIDToLabel = None):
        self.X = []
        self.Xnorm = None
        self.y = []
        self.yStrings = []
        self.imageNames = []

        self.numLabels = 0
        self.testAccuracy = -1.0
        self.count = 0
        self.name = name
        self.SetLabelDict(dictIDToLabel)
        self.pickleFileNameIn = pickleFileNameIn

        if (not pickleFileNameIn is None):
            self.ReadPickleFile(pickleFileNameIn)

    #--------------------------------- SetLabelDict
    def GetStatsStr(self):
        statsStr = "Dataset '{}' has {} items of shape {}".format(self.name, self.count, self.GetXFeatureShape())
        return statsStr

    #--------------------------------- SetLabelDict
    def SetLabelDict(self, dictIDToLabel):
        self.dictIDToLabel = dictIDToLabel
        if (self.dictIDToLabel is None):
            self.labelsStr = None
            self.labelsIDs = None
            self.numLabels = None
        else:
            self.labelsStr = self.dictIDToLabel.values()
            self.labelsIDs = self.dictIDToLabel.keys()
            self.numLabels = len(self.labelsIDs)

    #--------------------------------- ReadPickleFile
    def ReadPickleFile(self, pickleFileNameIn):
        with open(pickleFileNameIn, mode='rb') as fhandle:
            dictDS = pickle.load(fhandle)

            self.X, self.y = dictDS['features'], dictDS['labels']
            if (self.dictIDToLabel is not None):
                self.yStrings = [self.dictIDToLabel[yval] for yval in self.y]

            self.count = len(self.y)
            self.pickleFileNameIn = pickleFileNameIn
            imageNameBase = os.path.basename(pickleFileNameIn)
            self.imageNames = ["{}[{}]".format(imageNameBase, imageIndex) for imageIndex in range(self.count)]

    #--------------------------------- WritePickleFile
    def WritePickleFile(self, pickleFileNameOut):
        dictDS = {
            "features": self.X,
            "labels": self.y,
        }
        with open(pickleFileNameOut, 'wb') as handle:
            pickle.dump(dictDS, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #--------------------------------- Concatenate
    def Concatenate(self, dsOther):
        # self.X.concatenate(dsOther.X)
        self.X = np.concatenate([self.X, dsOther.X])
        self.y = np.concatenate([self.y, dsOther.y])
        self.yStrings += dsOther.yStrings
        self.imageNames += dsOther.imageNames
        self.count = len(self.y)

    #--------------------------------- GetXFeatureShape
    def GetXFeatureShape(self):
        return self.X[0].shape

    #--------------------------------- GetXFeatureShape
    def GetDSNumLabels(self):
        return len(set(self.y))

    #--------------------------------- DebugTruncate
    def DebugTruncate(self, truncSize):
        self.X = self.X[:truncSize]
        self.y = self.y[:truncSize]
        self.count = len(self.X)

    #--------------------------------- DebugTruncate
    def SegregateByLabel(self):
        print("\nSegregating {} images of dataset '{}' into {} datasets...".format(self.count, self.name, self.numLabels))
        timerStart = time.time()

        listDSSegregated = [CDataSet(name= "SegLabel{:02}".format(labelID), dictIDToLabel = self.dictIDToLabel) for labelID in self.labelsIDs]
        
        for index in range(self.count):
            cury = self.y[index]
            #print("image({:05})->y({:02})".format(index,cury), end='\r', flush=True)
            segSetCur = listDSSegregated[cury]

            segSetCur.X.append(self.X[index])
            segSetCur.y.append(self.y[index])
            segSetCur.yStrings.append(self.yStrings[index])
            segSetCur.imageNames.append(self.imageNames[index])
            segSetCur.count +=1

        for (dsIndex, segSetCur) in enumerate(listDSSegregated):
            segSetCur.X = np.array(segSetCur.X)
            segSetCur.y = np.array(segSetCur.y)

        timerElapsedS = time.time() - timerStart
        print("\nDone in {:.1f} seconds".format(timerElapsedS))

        return listDSSegregated

    #--------------------------------- MirrorSelfAppend
    def MirrorSelfAppend(self):
        print("Mirroring...", end='', flush=True)

        #flipY = 1
        #mirrorX = [cv2.flip(src=img, flipCode=flipY) for img in self.X]
        mirrorX = [np.fliplr(img) for img in self.X]
        mirrory = [-1 * val for val in self.y]

        self.X = np.concatenate([self.X, mirrorX])
        self.y = np.concatenate([self.y, mirrory])

        self.count = len(self.y)
        assert (len(self.y) == len(self.X))

        self.name += "_mirror"
        print("Done.")
        return (mirrorX, mirrory)

    #--------------------------------- CropInPlace
    def CropInPlace(self):
        print("Cropping...", end='', flush=True)
        croppedX = [img[0:100,:] for img in self.X]
        self.X = np.array(croppedX)
        self.name += "_crop"
        print("Done.")

    #--------------------------------- NormalizeTF
    def NormalizeTF(self, doConvertGray=False):
        import tensorflow as tf
        print("Normalizing...", end='', flush=True)

        if doConvertGray:
            Xin = tf.image.rgb_to_grayscale(self.X)
        else:
            Xin = self.X

        tfNorm = tf.map_fn(lambda img: tf.image.per_image_standardization(img),  Xin, dtype=tf.float32)

        with tf.Session() as sess:
           self.XNorm = tfNorm.eval()

        print("Done.")
        return self.XNorm

    #--------------------------------- Normalize
    def Normalize(self):
        print("Normalizing...", end='', flush=True)

        XNormNew = [((img / 127.5) - 1.0) for img in self.X]
        self.XNorm = XNormNew
        self.XNorm = np.array(XNormNew)
        print("Done.")
        return self.XNorm




############################################################################
############################ STANDALONE ####################################
############################################################################

# --------------------------------- ReadImageMetaDataCSV
def ReadImageMetaDataCSV(csvFileNameIn):
    dictCSVCols = {}
    dictCSVCols['imageFileNamesCenter'] = []
    dictCSVCols['steeringVals'] = []

    with open(csvFileNameIn, mode='r') as infile:
        infile.readline()  # Skip header line 0
        reader = csv.reader(infile)

        for (fnameCenter, fnameLeft, fnameRight, steeringVal, throttleVal, brakeVal, speedVal) in csv.reader(infile):
            dictCSVCols['imageFileNamesCenter'].append(fnameCenter)
            dictCSVCols['steeringVals'].append(float(steeringVal))

    return (dictCSVCols)


# --------------------------------- CreateDataSetFromImageFilesProj4
def CreateDataSetFromImageFilesProj4(csvFileNameIn, dataSetName, numInputLimit=0):
    #print("\nCreating dataset '{}' from file list: {}... ".format(dataSetName, csvFileNameIn), end='', flush=True)
    csvFilePath = os.path.dirname(csvFileNameIn)

    dsOut = CDataSet(name=dataSetName)

    dictCSVCols = ReadImageMetaDataCSV(csvFileNameIn)
    dsOut.y = np.array(dictCSVCols['steeringVals'])

    fileNamesIn = dictCSVCols['imageFileNamesCenter']
    images = []

    for imageIndex, imageFileNameIn in enumerate(fileNamesIn):
        imageFileNameBase = os.path.basename(imageFileNameIn)

        imageFileNameInFQ = csvFilePath +"/IMG/" + imageFileNameBase

        img = mpimg.imread(imageFileNameInFQ)
        images.append(img)
        dsOut.imageNames.append(imageFileNameBase)

        if (numInputLimit > 0 and imageIndex == (numInputLimit-1)):
            print("inputs TRUNCATED to {} items! ".format(numInputLimit), end='',flush=True)
            dsOut.y = dsOut.y[:numInputLimit]
            break

    dsOut.count = len(dsOut.y)
    dsOut.X = np.array(images)
    assert (len(dsOut.y) == len(dsOut.X))
    print("Finished reading {} image files.".format(dsOut.count))
    return (dsOut)

#--------------------------------- CreateDataSetFromImageFiles
def CreateDataSetFromImageFilesProj33(dirIn, dictIDToLabel, dataSetName="jpgFiles"):
    imageRecs = []
    fileExt = '*.jpg'
    fileSpec = dirIn + fileExt
    print("\nCreating dataset '{}' from files: {}... ".format(dataSetName, fileSpec), end='', flush=True)

    fileNamesIn = glob.glob(fileSpec)
    reExtractLabelNum = "(\d\d)\."

    dsOut = CDataSet(name=dataSetName, dictIDToLabel=dictIDToLabel)
    images = []

    for fileNameIn in fileNamesIn:
        labelId = labelIdStr = None
        reResult = re.search(reExtractLabelNum, fileNameIn)
        if reResult:
            labelId = int(reResult.groups()[0])
            labelStr = dictIDToLabel[labelId]

        fileNameBase = os.path.basename(fileNameIn)
        fileNameInFQ = os.path.abspath(fileNameIn)

        img =  mpimg.imread(fileNameInFQ)
        images.append(img)

        dsOut.y.append(labelId)
        dsOut.yStrings.append(labelStr)
        dsOut.imageNames.append(fileNameBase)

    dsOut.count = len(dsOut.y)
    dsOut.X = np.array(images)
    if (g_ConvertImagesToTensors):
        dsOut.X = tf.image.convert_image_dtype(dsOut.X, dtype=tf.float32)

    print("finished reading {} image files.".format(dsOut.count))
    return(dsOut)

#--------------------------------- ReadTrainingSets
def ReadTrainingSets(training_file, validation_file, testing_file, dictIDToLabel=None, truncatedTrainingSetSize=0):

    print("\nReading Training sets...")
    dsTrainRaw = CDataSet(name="TrainRaw", pickleFileNameIn = training_file, dictIDToLabel = dictIDToLabel)
    dsValidRaw = CDataSet(name="ValidRaw", pickleFileNameIn = validation_file, dictIDToLabel = dictIDToLabel)
    dsTestRaw  = CDataSet(name="TestRaw", pickleFileNameIn = testing_file, dictIDToLabel = dictIDToLabel)

    if truncatedTrainingSetSize > 0:
        print("*************** WARNING: Debug data sets truncated to size {}! *****************".format(truncatedTrainingSetSize))
        dsTrainRaw.DebugTruncate(truncatedTrainingSetSize)
        dsValidRaw.DebugTruncate(truncatedTrainingSetSize)
        dsTestRaw.DebugTruncate(truncatedTrainingSetSize)

    print("Raw training set size (train, validation, test) =({}, {}, {})".format(dsTrainRaw.count, dsValidRaw.count, dsTestRaw.count))
    print("Image data shape =", dsTrainRaw.GetXFeatureShape())
    print("Number of classes =", dsTrainRaw.GetDSNumLabels())

    if (g_ConvertImagesToTensors):
        dsTrainRaw.X = tf.image.convert_image_dtype(dsTrainRaw.X, dtype=tf.float32)
        dsValidRaw.X = tf.image.convert_image_dtype(dsValidRaw.X, dtype=tf.float32)
        dsTestRaw.X = tf.image.convert_image_dtype(dsTestRaw.X, dtype=tf.float32)

    return dsTrainRaw, dsValidRaw, dsTestRaw

#--------------------------------- ReadLabelDict
def ReadLabelDict(signLabelsCSVFileIn):
    '''
    Create dictionaries from the csv. Assumes line 0 header and unique indices in order
    '''

    with open(signLabelsCSVFileIn, mode='r') as infile:
        infile.readline() # Skip header line 0
        reader = csv.reader(infile)
        dictIDToLabel = OrderedDict( (int(row[0]), row[1]) for row in csv.reader(infile) )
    return dictIDToLabel



#====================== Main() =====================
def Main(gArgs):
    dsTraining = CreateDataSetFromImageFilesProj4(gArgs.signLabelsCSVFileIn, dataSetName="augTest", numInputLimit=gArgs.truncatedTrainingSetSize)
    print("DSINfo:", dsTraining.GetStatsStr())
    dsTraining.MirrorSelfAppend()
    print("DSINfo:", dsTraining.GetStatsStr())
    dsTraining.CropInPlace()
    print("DSINfo:", dsTraining.GetStatsStr())
    dsTraining.Normalize()

#--------------------------------- GetgArgs
def GetgArgs():
    gArgs = type("GlobalArgsContainer", (object,), {})

    gArgs.numEpochs = 5
    gArgs.batchSize = 128
    gArgs.trainRate = 0.001

    gArgs.doConvertGray = True

    # For debug testing only
    gArgs.truncatedTrainingSetSize = 100  # Default: 0 (Use full training sets). Truncation is for speedup of debug cycle
    gArgs.doTrainModel = True
    gArgs.doSaveModel = True

    gArgs.doComputeAugments = True  # Otherwise loads from file
    gArgs.doSaveTrainCompleteFile = False

    # I/O Directories
    gArgs.trainingFilesDirIn = "./Assets/trainingdata/"
    gArgs.finalTestFilesDirIn = "./Assets/finalTest/"
    gArgs.sessionSavesDir = "./Assets/sessionSaves/"
    gArgs.modelSavesDir = "./Assets/modelSaves/"

    # Input files
    gArgs.sessionSaveFile = gArgs.sessionSavesDir + "session"
    gArgs.modelSaveFile = gArgs.modelSavesDir + "model.h5"

    gArgs.signLabelsCSVFileIn = gArgs.trainingFilesDirIn + "driving_log.csv"
    gArgs.trainingFileIn = gArgs.trainingFilesDirIn + "train.p"
    gArgs.validationFileIn = gArgs.trainingFilesDirIn + "valid.p"
    gArgs.testingFileIn = gArgs.trainingFilesDirIn + "test.p"
    gArgs.trainingCompleteFile = gArgs.trainingFilesDirIn + "trainComplete.p"

    return gArgs


#====================== Main Invocation =====================
if ((__name__ == '__main__')):
    Main(GetgArgs())