#from keras.utils.vis_utils import plotting_tools
#import keras.utils.vis_utils.

import matplotlib
import matplotlib.image as mpimg

#----------------------- PlottingBackend_Switch()
def PlottingBackend_Switch(whichBackEnd):
    import matplotlib
    matplotlib.use(whichBackEnd, warn=False, force=True)
    from matplotlib import pyplot as plt
    print("Switched to:", matplotlib.get_backend())

PlottingBackend_Switch('TkAgg') # inline notebook GTK GTKAgg GTKCairo GTK3Agg GTK3Cairo Qt4Agg Qt5Agg TkAgg WX WXAgg Agg Cairo GDK PS PDF SVG
import matplotlib.pyplot as plt
#plt.ioff()  # turns on interactive mode

#----------------------- TFVerbosity()
def TFVerbosity(verbosityLevel = 3):
    """
    Lower versbosity of tensorflow
    :param verbosityLevel:
    :return:
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
    tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, or FATAL

TFVerbosity(verbosityLevel=3)

#----------------------- checkGPU()
def checkGPU():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
checkGPU()

#====================== GLOBALS ==================
def GetgArgs():
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
    gArgs.doTrainModel = True
    gArgs.doSaveModel = True

    #gArgs.doComputeAugments = False  # Otherwise loads from file
    gArgs.trainingDataSource = "raw" # pickle or raw
    #gArgs.doSaveTrainCompleteFile = True

    # I/O Directories
    gArgs.trainingFilesDirIn = "./Assets/trainingdata/"
    gArgs.finalTestFilesDirIn = "./Assets/finalTest/"
    gArgs.sessionSavesDir = "./Assets/sessionSaves/"
    gArgs.modelSavesDir = "./Assets/modelSaves/"

    # Input files
    gArgs.sessionSaveFile = gArgs.sessionSavesDir + "session"
    #gArgs.modelSaveFile = gArgs.modelSavesDir + "modelAug.h5"
    gArgs.modelSaveFileNameBase = "model"
    gArgs.modelLoadFileNameBase = "model_nvidia_AugM_2018-10-09T14-41-50"

    gArgs.csvFileNameIn = gArgs.trainingFilesDirIn + "driving_log.csv"

    #gArgs.trainingFileIn = gArgs.trainingFilesDirIn + "train.p"
    #gArgs.validationFileIn = gArgs.trainingFilesDirIn + "valid.p"
    #gArgs.testingFileIn = gArgs.trainingFilesDirIn + "test.p"
    #gArgs.trainingCompleteFile = gArgs.trainingFilesDirIn + "trainComplete.p"
    #gArgs.trainingCompleteFile = gArgs.trainingFilesDirIn + "trainCompleteAug.p"
    #gArgs.trainingCompleteFile = gArgs.trainingFilesDirIn + "trainGenAug.p"

    gArgs.doShowPlots = True

    dt = datetime.datetime.now()
    gArgs.strDT = "_{:%Y-%m-%dT%H-%M-%S}".format(dt)

    return gArgs


#
#if np.isnan(np.sum(imgCur)) or np.isnan(steerAngle):
#    print('NP NaN!!!')
#    raise ValueError('NP NaN!!!')
