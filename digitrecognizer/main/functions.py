import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
%matplotlib inline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report
import os, random
import cv2
from glob import glob
import sklearn
from sklearn.model_selection import train_test_split

 

def predict(request):
    model=tf.keras.models.load_model("./anoth_model11.h5")
    result=0
    imgData = request.POST.get('img')
    img = np.asarray(imgData)
    img=img.reshape(1,28,28,1)
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    
    if probabilityValue > 0.3:
        result=(classIndex[0])
    else:
        result=(0)
    return (result)

    return result