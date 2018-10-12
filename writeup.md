# carND Proj4: Behavioral Cloning
### Submission writeup: ChrisL 2018-10-12<br/>
**Notes to Reviewer:**<br/>
Using Keras 2.0.9. See '_Important Environment Note_' below
---

## **Project Overview**
The goals / steps of this project are the following:
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
    
---

## Files Submitted
**Important Environment Note: Keras Version** <br/> 
My development environment has Keras 2.0.9.
I believe that the lessons and VM may have used earlier versions.
Therefore there is a chance that model.py won't run in the 
'standard' environment and further that the model won't properly
load for running the sim client.


#### 1. The files of interest:
All in repo root.
* [model.py](./model.py)<br/> 
The main file contains the model(s) implementation, the training code.
All default runtime parameters are established ```def GetgArgs():``` in this file.
The ```def Main(gArgs):``` is the top level function for 
managing the training process. 

* [trainrecs.py](./model.py)<br/> 
This file contains functions for managing training datasets
and X[] y[] generators for the keras fitting function.

* [model.h5](./model.h5)<br/>
This was the best model that I was able to generate.
It is launched with ```python drive.py model.h5```

* [video.mp4](./model.h5)<br/>
This is a recording of a single counter clockwise lap,
using the submitted model.h5.

* [drive.py](./drive.py) **REQUIRED!**<br/> 
This file is the Sim control client that uses the trained model.
It is required for using my trained model because it has been
modified to handle image cropping explicitly before
before passing into the network.<br/> 
It has been quite helpful to see a simple python client/server architecture
implementation.

 ---
Files not of interest:<br/> 

[./notused/ Directory](./notused/)<br/> 
This folder contains files that have been abandoned
as far as the submission, but may be revived for future
use:<br/> 
[cdataset.py](./notused/cdataset.py):
A frankenstein file from previous projects that Implements a dataset container object.
[utils.py](./notused/utils.py):
Some extraneous utilities removed from model.py for submission.
[snippets.py](./notused/snippets.py):
Code from the lessons and abandoned code.
[model_training.ipynb](./notused/model_training.ipynb):
A Dev/debug notebook.


## Implementation Notes

### 1. Model Architecture
I tried several increasingly elaborate CNNs, following along with the classroom lessons.
The implementation for each of these is clear to read in model.py in the functions
```CreateModelBasic(), CreateModelLenet(), and CreateModelNvidia()```
The code is mostly directly borrowed from the youtube videos. I tried each of these models
and found that the [Nvidia model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
, designed for this purpose, delivered minimum loss in a very small number of epochs.
I only road tested one run each from the first 2 models and found that driving off the road
was common even in easy areas.

Here is the architecture layer summary of the NVidia network that I used
for the final driving solution. <br/>
The preprocessings consists of cropping out the top and bottom of the image
that rarely have useful roadway image information followed by standard normalization
of pixel values to -1.0 to +1.0 as standard practice.
Then 5 convolutional layers that whose deep layer is flattened and fed to a 4 layer fully connected 
network of with relu activation and dropout layers and finally a single output neuron that is interpreted as the 
predicted steering angle.
I originally used 5 fully connected layers, as in the lessons, but found that
the number of training parameters was enormous, so I dropped one of them.


|Layer (type)                | Output Shape       | Param #   |
| -------------------------- | ------------------ | --------- |
|cropping2d_1 (Cropping2D)   | (90, 320, 3)       | 0         |
|lambda_1 (Normalize)        | (90, 320, 3)       | 0         |
|conv2d_1 (Conv2D)           | (43, 158, 24)      | 1824      |
|conv2d_2 (Conv2D)           | (20, 77, 36)       | 21636     |
|conv2d_3 (Conv2D)           | (8, 37, 48)        | 43248     |
|conv2d_4 (Conv2D)           | (6, 35, 64)        | 27712     |
|conv2d_5 (Conv2D)           | (4, 33, 64)        | 36928     |
|flatten_1 (Flatten)         | (8448)             | 0         |
|dense_1 (Dense)             | (100)             | 8449000   |
|dropout_1 (Dropout)         | (100)             | 0         |
|dense_2 (Dense)             | (50)              | 5050    |
|dropout_2 (Dropout)         | (50)              | 0         |
|dense_3 (Dense)             | (10)               | 510      |
|dropout_3 (Dropout)         | (10)               | 0         |
|dense_4 (Dense)             | (1)               | 11       |
```
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
```

With Keras it is very easy to see the  layer structure by visual inspection of
the source code:

```
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

```

#### 3. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually. 
The batch size was left at 128, as larger sizes run into GPU memory limits 
on my computer.

#### 4. Attempts to reduce overfitting in the model
In earlier models I found than after many epochs the training loss continued to drop
while the test loss did not continue to improve suggesting that there was overfitting.
 
![LossRate](./Assets/writeupImages/LossPlot15Epochs_LowDropout.png)
 
In order to mitigate overfitting I added dropout layers between each fully connected layer.
This seemed to cause the training loss to stablize.
![LossRate](./Assets/writeupImages/LossPlot5Epochs_MedDropout.png)

I found that by adjusting those dropout rates to around 30% 
I was able to get training and validation loss to approach each other, rather than cross, and
with still a perceivable, if slow, decline in validation loss

#### 4. Training Data & Augmentation
**Training data source**<br/>
I used exclusively the training data set for the easy track provided by udacity. With more time
I would generate more training data: Counter clockwise laps, difficult area runs and
bad car orientation recoveries would all benefit the training.
I did not capture training data for the hilly jungle track or ever attempt to
train or run it.

**Cropping**<br/>
Since the upper portion of the training images contained
irrelevant features like trees and sky and the lower portion contains only pavement
I added a preproccessing step to remove the top 60 and bottom 10 pixels from
every image. 

Originally I used a keras cropping layer, which improved processing speed
but I switched to manual preproccesing (in trainrecs.py::TrainRecordBatchGenerator())
when I was trying to solve a NaN loss training problem. 
This means that the sim client drive.py also needed to perform the same preprocessing
crop before submitting the image to the trained network. In future revisions
I would plan to restore this keras cropping layer.

**Mirroring**<br/>
I augmented the center cam dataset images with mirror images and substituted
the negative of the correponding steering angle for those images. The reasoning being the 
the essential features of the roadway and the correct steering response should be symmetric 
(but maybe not on real hightways that have asymmetic markings!)
This is accomplished in model.py::Main() by making a copy center cam training records
with the steering angle negated and the doMirror flag set. 
```
# Note the "-" and "not". This is where the mirror field are altered. The image itself is flipped in the generator.
trainRecsCenterMirror = [trecs.CTrainingRecord(rec.fileName, -rec.steeringAngle, rec.whichCam, not rec.doMirror) for rec in trainRecsCenterNoMirror]
```
The actual image flip is deferred until the image is retrieved from disk by the generator in 
trainrecs.py::TrainRecordBatchGenerator() at the call to 
```
imgCur = trainRecCur.GetImage() # Takes care of mirror internally, if needed
```
Here is a sample, raw center cam image:<br/>

![SampleCropMirror](./Assets/writeupImages/center_2016_12_01_13_46_21_144.jpg)

Here is that sample, not mirrored cropped and mirrored, both cropped:<br/>

![SampleCropMirror](./Assets/writeupImages/MirrorCrop.png)
**Side Cams**<br/>
Another augmentation of the dataset was achieved by using the side camera images. Since these
images have a different perspective than the center cam the correct steering angle needed to 
be adjusted. I adjusted the steering angle by +- 0.25 degrees compared to the center cam
true steering angle. Larger corrections caused very erratic steering and lesser corrections
didn't caused worse performance than just training with the center cam dataset.
I used only a portion of these images (half the number of center cam images) in the training set 
- I did not want to overwhelm
the training with guessed data.

Here are the Left, Center and Right images from the same moment:<br/>

![SampleCropMirror](./Assets/writeupImages/rawLeft.png)
![SampleCropMirror](./Assets/writeupImages/rawCenter.png)
![SampleCropMirror](./Assets/writeupImages/rawRight.png)

**Training Validation Split**<br/>
For the training validation test I split off 20% of the _center cam only_, 
because the side cams with their steering angle correction seemed like
a potentially erroneous dataset to validate against. I am uncertain if this
is the right approach. Especially since the test loss was consistenly lower than
the training loss. However since the test loss on 'real' data was decreasing steadily through the
epochs I believe that it does demonstrate that the training was improving.

#### 5. Improvements TODO
Potential improvements have been noted in the implementation notes.
 
#### 6. Difficulties Encountered
I encountered a major, time consuming problem just before submission where
training started to sometiomes compute NaN loss values. I eventually traced this
to the way my generator was creating numpy arrays. This problem did however
spur me to create a much cleaner training record management and generator
system.

As usual the dev/debug cycle was time consuming because training took a long
time on my computer/GPU. Additionally, I found that system suspends would cause
the GPU to malfunction, but I couldn't risk the potential chaos of updating
the GPU driver or other fixes, so I had to live with it for now.

## Links
[ProjectRubric](https://review.udacity.com/#!/rubrics/432/view)<br/>

[My Project Repo](https://github.com/cielsys/CarNDProj4_CarBehave)<br/>
This, my main project repo.

[Udacity Main Project Template repo](https://github.com/udacity/CarND-Behavioral-Cloning-P3)<br/>
[TrainingData](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)<br/>

[SimulatorRepo](https://github.com/udacity/self-driving-car-sim)<br/>
[SimBinary_Linux64_2017-02-07](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)<br/>

[//]: # (Image References)
[//]: # (html resizeable image tag <img src='./Assets/examples/placeholder.png' width="480" alt="Combined Image" />)
