import os
import errno
import csv
import random
import math
import datetime
import pickle

import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Dense, Flatten, Lambda
from keras.layers import Cropping2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint
from keras import optimizers

import sklearn
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# local imports
import trainrecs as trecs

#np.random.seed(44) # I want to see same results everytime for debug

#====================== GLOBALS ==================
def GetgArgs():
    '''
    Provides a container with all the relevant runtime default
    values for training and fileIO.
    :return:
    '''
    gArgs = type("GlobalArgsContainer", (object,), {})

    gArgs.modelType = "nvidia"
    gArgs.numEpochs = 5
    gArgs.batchSize = 256
    #gArgs.trainRate = 0.002

    #gArgs.doConvertGray = True
    gArgs.doAugments = False
    gArgs.camAngleCorrection = 0.25
    gArgs.imageShapeRaw = (160, 320, 3)
    gArgs.cropTBLR = (60, 10, 0, 0)
    gArgs.imageShapeCrop =(gArgs.imageShapeRaw[0] - (gArgs.cropTBLR[0] + gArgs.cropTBLR[1]),
                            gArgs.imageShapeRaw[1] - (gArgs.cropTBLR[2] + gArgs.cropTBLR[3]),
                            3)

    # For debug testing only
    gArgs.truncatedTrainingSetSize = 0  # Default: 0 (Use full training sets). Truncation is for speedup of debug cycle
    gArgs.doTrainModel = True # If not doTrain then will try to load gArgs.modelLoadFileNameBase model
    gArgs.doSaveModel = True

    # I/O Directories
    gArgs.trainingFilesDirIn = "./Assets/trainingdata/"
    gArgs.finalTestFilesDirIn = "./Assets/finalTest/"
    gArgs.sessionSavesDir = "./Assets/sessionSaves/"
    gArgs.modelSavesDir = "./Assets/modelSaves/"

    # Input files
    gArgs.sessionSaveFile = gArgs.sessionSavesDir + "session"
    gArgs.modelSaveFileNameBase = "model"
    gArgs.modelLoadFileNameBase = "model_nvidia_2018-10-12T06-56-51" # Good One!

    gArgs.csvFileNameIn = gArgs.trainingFilesDirIn + "driving_log.csv"
    gArgs.doShowPlots = True

    # Per run time stamp for saving various files
    dt = datetime.datetime.now()
    gArgs.strDT = "_{:%Y-%m-%dT%H-%M-%S}".format(dt)

    return gArgs

#====================== CODE =====================

#--------------------------------- PlotTrainingHistory()
def PlotTrainingHistory(model, dictHistory):
    from keras.utils.vis_utils import plot_model
    plot_model(model)

    # summarize history for loss
    plt.plot(dictHistory['loss'])
    plt.plot(dictHistory['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#--------------------------------- SaveModel()
def SaveModel(model, dictHistory, modelDir, fileNameBase="model"):
    fileNameExt = ".h5"
    fileNameHistoryExt = ".trainhistory.p"

    fileNameFQ = os.path.abspath( modelDir + fileNameBase + fileNameExt)
    fileNameHistoryFQ = os.path.abspath( modelDir + fileNameBase + fileNameHistoryExt)

    if not os.path.exists(os.path.dirname(fileNameFQ)):
        os.makedirs(os.path.dirname(fileNameFQ))

    print("Saving model file {}".format(fileNameFQ))
    model.save(fileNameFQ)

    with open(fileNameHistoryFQ, 'wb') as fhHist:
        pickle.dump(dictHistory, fhHist)

#--------------------------------- LoadModel()
def LoadModel(modelDir, fileNameBase):
    fileNameExt = ".h5"
    fileNameHistoryExt = ".trainhistory.p"

    fileNameFQ = os.path.abspath( modelDir + fileNameBase + fileNameExt)
    fileNameHistoryFQ = os.path.abspath( modelDir + fileNameBase + fileNameHistoryExt)

    print("Reading model file {}".format(fileNameFQ))
    model = load_model(fileNameFQ)

    with open(fileNameHistoryFQ, 'rb') as fhHist:
        dictHistory = pickle.load(fhHist)

    return(model, dictHistory)

#--------------------------------- CreateModel()
def CreateModelBasic(gArgs, shapeIn):
    '''Obsolete.'''
    model = Sequential()

    model.add(Lambda(lambda img: img/127.5 - 1.0, input_shape=shapeIn))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(128))
    #model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    return(model)

#--------------------------------- CreateModel()
def CreateModelLenet(gArgs, shapeIn, cropVH=None):
    '''Obsolete.'''
    model = Sequential()
    model.add(Cropping2D(cropping=cropVH, input_shape=shapeIn))
    model.add(Lambda(lambda img: img/127.5 - 1.0))
    model.add(Conv2D(6,(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6,(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(0.50))
    model.add(Dense(84))
    model.add(Dense(1))

    return(model)

#################################### ACTIVE MODEL ############################
#--------------------------------- CreateModelNvidia()
def CreateModelNvidia(gArgs):
    model = Sequential()

    #model.add(Cropping2D(cropping=cropVH, input_shape=shapeIn)) # Cropping handled in trecs.TrainRecordBatchGenerator()
    #model.add(Lambda(lambda img: img/127.5 - 1.0))
    model.add(Lambda(lambda img: img/127.5 - 1.0, input_shape=gArgs.imageShapeCrop)) # Normalization

    # Covolutional Layers
    model.add(Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))

    # Fully Connected layers
    model.add(Flatten())
    #model.add(Dense(1164, activation='relu')) # Removed: Training param explosion!
    #model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='tanh'))

    return model

#====================== Main() =====================
def Main(gArgs):

    #################### PREPARE TRAINING SETS
    csvRowsRaw = trecs.ReadCSVFile(gArgs.csvFileNameIn, limitSize=gArgs.truncatedTrainingSetSize)
    trainRecsSidesNoMirror, trainRecsCenterNoMirror = trecs.CSVRawRowsToTrainingRecs(csvRowsRaw, camAngleCorrection=gArgs.camAngleCorrection)

    # Build records for the mirrored version from only Center cam images.
    # Note the "-" and "not". This is where the mirror field are altered. The image itself is flipped in the generator.
    trainRecsCenterMirror = [trecs.CTrainingRecord(rec.fileName, -rec.steeringAngle, rec.whichCam, not rec.doMirror) for rec in trainRecsCenterNoMirror]
    trainRecsCenterBoth = trainRecsCenterNoMirror + trainRecsCenterMirror
    np.random.shuffle(trainRecsCenterBoth)

    trainRecsSidesMirror = [trecs.CTrainingRecord(rec.fileName, -rec.steeringAngle, rec.whichCam, not rec.doMirror) for rec in trainRecsSidesNoMirror]
    trainRecsSidesBoth = trainRecsSidesNoMirror # This seems to cause trouble + trainRecsSidesMirror
    np.random.shuffle(trainRecsSidesBoth)

    # Split off validation set from only Center cam images
    trainRecsCenterSplit, validRecsCenter = train_test_split(trainRecsCenterBoth, test_size=0.2)

    # Add the side cams to the full training set
    trainRecsFull = trainRecsCenterSplit + trainRecsSidesBoth[:len(trainRecsCenterSplit)//2]
    np.random.shuffle(trainRecsFull)
    #trecs.PrintRecords(validRecsCenter, numShow=100)

    # Create generators
    generTrain = trecs.TrainRecordBatchGenerator(trainRecsFull, batchSize = gArgs.batchSize, cropTBLR=gArgs.cropTBLR)
    generValid = trecs.TrainRecordBatchGenerator(validRecsCenter, batchSize = gArgs.batchSize, cropTBLR=gArgs.cropTBLR)

    numSampleTrain, numSampleValid = len(trainRecsFull), len(validRecsCenter)
    numBatchesPerEpochTrain = int(math.ceil(numSampleTrain / gArgs.batchSize))
    numBatchesPerEpochValid = int(math.ceil(numSampleValid / gArgs.batchSize))



    #################### CREATE MODEL
    if gArgs.doTrainModel:
        print("Creating Model...")
        model = CreateModelNvidia(gArgs)

        print("Compiling Model...")
        # optimizer = optimizers.Adam(clipnorm=1.0, clipvalue=0.5)
        optimizer = optimizers.Adam()
        model.compile(optimizer=optimizer, loss='mse')


        #################### TRAIN MODEL #########################################################
        print("Training Model: Epochs={}, trainSetSize={}".format(gArgs.numEpochs, numSampleTrain))

        # checkptFmtStr = gArgs.modelSavesDir + "model" + strDT +"_{epoch:03d}.h5"
        # cbCheckpoint = ModelCheckpoint(checkptFmtStr, monitor='val_loss',verbose=0, save_best_only=True, mode='auto')
        # cbPlot = plotting_tools.LoggerPlotter()
        # callbacks = [cbCheckpoint] #[cbPlot]
        # metricsList = ['accuracy']
        # workers = 2
        # callbacks=callbacks) #workers = workers)
        history = model.fit_generator(epochs=gArgs.numEpochs,
                                      generator=generTrain, steps_per_epoch=numBatchesPerEpochTrain,
                                      validation_data=generValid, validation_steps=numBatchesPerEpochValid)

        #################### SAVE MODEL
        if (gArgs.doSaveModel):
            dictHistory = history.history

            fileNameBase = "{}_{}{}".format(gArgs.modelSaveFileNameBase, gArgs.modelType,  gArgs.strDT)
            SaveModel(model, dictHistory, gArgs.modelSavesDir, fileNameBase)
        else:
            print("Model save DISABLED")


    else:
        #################### LOAD MODEL
        model, dictHistory = LoadModel(gArgs.modelSavesDir, gArgs.modelLoadFileNameBase)



    #################### PLOT & EVAL
    if gArgs.doShowPlots:
        print(model.summary())
        PlotTrainingHistory(model, dictHistory)

        generTemp = trecs.TrainRecordBatchGenerator(trainRecsCenterBoth, batchSize = gArgs.batchSize, cropTBLR=gArgs.cropTBLR)
        XBatch, yBatch = next(generTemp)

        yPredicted = np.transpose(model.predict(XBatch))[0]
        print("ErrMin", np.amin(yPredicted - yBatch))
        print("ErrMax", np.amax(yPredicted - yBatch))
        print("ErrAvg", np.mean((yPredicted - yBatch) ** 2))


#====================== Main Invocation =====================
if ((__name__ == '__main__')):
    Main(GetgArgs())