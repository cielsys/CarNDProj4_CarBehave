import pickle
import numpy as np
import tensorflow as tf

# Split the data
X_train, y_train = data['features'], data['labels']

# Load pickled data
with open('small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
max

input_shape=(32, 32, 3)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
#1st Layer - Add a flatten layer
model.add(Flatten())
#2nd Layer - Add a fully connected layer
model.add(Dense(128))
#3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))
#4th Layer - Add a fully connected layer
model.add(Dense(5))
#5th Layer - Add a ReLU activation layer
model.add(Activation('softmax'))

# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)


# compile and fit the model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, epochs=22, validation_split=0.2)

# evaluate model against the test data
with open('small_test_traffic.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
X_normalized_test = np.array(X_test / 255.0 - 0.5 )
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")

metrics = model.evaluate(X_normalized_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))


    # history = model.fit(dsTraining.X, dsTraining.y, batch_size=gArgs.batchSize, epochs=gArgs.numEpochs, validation_split=0.2, shuffle=True)

###############################################################################
GENERATOR
###############################################################################
import os
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)



#====================== MainDirect() =====================
def MainDirect(gArgs):

    #################### READ DATASETS
    if (gArgs.trainingDataSource == 'raw'):
        dsTraining = CreateDataSetFromImageFilesProj4(gArgs.signLabelsCSVFileIn, dataSetName="udaTraining", numInputLimit=gArgs.truncatedTrainingSetSize)
        dsTraining.MirrorSelfAppend()
    elif(gArgs.trainingDataSource == 'pickle'):
        print("Reading training pickle file from {}.".format(gArgs.trainingCompleteFile))
        dsTraining = CDataSet("trainComplete", gArgs.trainingCompleteFile)

    if (gArgs.trainingDataSource != 'pickle') and not gArgs.doSaveTrainCompleteFile:
        print("Writing training pickle file to {}.".format(gArgs.trainingCompleteFile))
        dsTraining.WritePickleFile(gArgs.trainingCompleteFile)

    print("DSINfo:", dsTraining.GetStatsStr())


    if gArgs.doTrainModel:
        cropTopBottom = (60, 10)
        cropLeftRight = (0, 0)
        cropVH=(cropTopBottom, cropLeftRight)
        shapeIn = dsTraining.GetXFeatureShape()

        #################### CREATE MODEL
        print("Creating Model...")
        model = CreateModelNvidia(gArgs, shapeIn, cropVH)

        #################### NORM, TRAINING
        metricsList = ['accuracy']
        print("Compiling Model...")
        model.compile(optimizer='adam', loss='mse', metrics=metricsList)

        print("Training Model using '{}'...".format(dsTraining.name))
        history = model.fit(dsTraining.X, dsTraining.y, epochs=gArgs.numEpochs, validation_split=0.2, shuffle=True)
        #history = model.fit(dsTraining.X, dsTraining.y, batch_size=gArgs.batchSize, epochs=gArgs.numEpochs, validation_split=0.2, shuffle=True)

        #################### PLOT & EVAL
        PlotTrainingHistory(history)
        yPredicted = np.transpose(model.predict(dsTraining.X))[0]

        if (gArgs.doSaveModel):
            SaveModel(model, gArgs.modelSaveFile)
        else:
            print("Model save DISABLED")



#--------------------------------- TrainRecordBatchGenerator()
def TrainRecordBatchGenerator(recRowsAll, imageShape, batch_size, doAugments):
    doConvertToHSV = False
    num_samples = len(recRowsAll)

    genReset=0
    while True: # Loop forever so the generator never terminates
        #print("In genreset={}".format(genReset))
        np.random.shuffle(recRowsAll)

        for offset in range(0, num_samples, batch_size):
            recRowsBatch = recRowsAll[offset:offset + batch_size]

            #XBatch = np.empty([batch_size, imageShape[0]-70, imageShape[1], imageShape[2]])
            #yBatch = np.empty(batch_size)
            XBatch = []
            yBatch = []
            for batchItemIndex, recRowCur in enumerate(recRowsBatch):
                (fnameCenter, fnameLeft, fnameRight, steeringVal, throttleVal, brakeVal, speedVal) = recRowCur

                imageSelector = 0#np.random.choice(3) if doAugments else 0
                if imageSelector == 0:
                    imgCur = mpimg.imread(fnameCenter)
                    steerAngle = float(steeringVal)
                elif imageSelector == 1:
                    imgCur = mpimg.imread(fnameLeft)
                    steerAngle = float(steeringVal) + 0.2
                else:
                    imgCur = mpimg.imread(fnameRight)
                    steerAngle = float(steeringVal) - 0.2

                if doConvertToHSV:
                    imgCur = matplotlib.colors.rgb_to_hsv(imgCur)

                if doAugments:
                    doMirror = random.random() > 0.5
                    if doMirror:
                        imgCur = np.fliplr(imgCur)
                        steerAngle = -steerAngle

                if np.isnan(np.sum(imgCur)) or np.isnan(steerAngle):
                    print('NP NaN!!!')
                    raise ValueError('NP NaN!!!')

                XBatch.append(imgCur[60:-10, :, :])
                yBatch.append(steerAngle)
                #XBatch[batchItemIndex]= imgCur[60:-10, :, :]
                #yBatch[batchItemIndex]= steerAngle

            # trim image to only see section with road
            XBatch = np.array(XBatch)
            yBatch = np.array(yBatch)
            yield (XBatch, yBatch)

            # --------------------------------- GetImageShapeFromCSVRecs()
            def GetImageShapeFromCSVRecs(csvRows):
                # generTemp = TrainRecordBatchGenerator(csvRows, batch_size=1, doAugments=True)
                # XBatch, yBatch  = next(generTemp)
                imageShape = (160, 320, 3)
                return (imageShape)


#--------------------------------- ReadImageMetaDataCSVToRowRecords()
def ReadImageMetaDataCSVToRowRecords(csvFileNameIn, limitSize=0):
    print("Creating dataset metadata from: '{}'... ".format(csvFileNameIn), end='', flush=True)
    csvFilePath = os.path.dirname(csvFileNameIn)
    imageDir = csvFilePath + "/IMG/"

    recRowsAll = []
    numRows = 0
    with open(csvFileNameIn, mode='r') as infile:
        infile.readline()  # Skip header line 0
        reader = csv.reader(infile)

        for recRowCur in csv.reader(infile):
            (fnameCenter, fnameLeft, fnameRight, steeringVal, throttleVal, brakeVal, speedVal) = recRowCur
            recRowCur[0] = imageDir + os.path.basename(fnameCenter)
            recRowCur[1] = imageDir + os.path.basename(fnameLeft)
            recRowCur[2] = imageDir + os.path.basename(fnameRight)
            recRowsAll.append(recRowCur)

            numRows+=1
            if limitSize == numRows:
                break

    np.random.shuffle(recRowsAll)
    print("Done.")
    return (recRowsAll)


    #################### READ DATASETS
    csvRowsAll = ReadImageMetaDataCSVToRowRecords(gArgs.signLabelsCSVFileIn, limitSize=gArgs.truncatedTrainingSetSize)
    csvRowsTrain, csvRowsValid = train_test_split(csvRowsAll, test_size=0.1)


    numSampleTrain, numSampleValid = len(csvRowsTrain), len(csvRowsValid)

    numTrainTotal = gArgs.augFactor * numSampleTrain

    generTrain = TrainRecordBatchGenerator(csvRowsTrain, imageShape, batch_size=gArgs.batchSize, doAugments=gArgs.doAugments)
    generValid = TrainRecordBatchGenerator(csvRowsValid, imageShape, batch_size=gArgs.batchSize, doAugments=False)


model, history = CreateTrainModel(gArgs, imageShape, generTrain, numTrainTotal, generValid, numSampleValid)

#--------------------------------- TestFunc()
def TestFunc(gArgs):
    print("In TESTFUNC...")
    model = "blah"
    if gArgs.doTrainModel:
        if (gArgs.doSaveModel):
            fileNameBase = "{}_{}_AugM".format(gArgs.modelSaveFileNameBase, gArgs.modelType)
            SaveModel(model, gArgs.modelSavesDir, fileNameBase)
        else:
            print("Model save DISABLED")
