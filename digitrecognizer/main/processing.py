from django.shortcuts import render
import numpy as np # linear algebra
import cv2,os
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
#from . import anoth_model11 as model
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
def main_outline(contour):
    biggest = np.array([])
    max_area = 0

    # We go through all the contours ans select the one 
    # that is largest and is a square
    for i in contour:

        # area of contour
        area = cv2.contourArea(i)
        if area >50:

            # perimeter
            peri = cv2.arcLength(i, True)

            # sides of polygon
            approx = cv2.approxPolyDP(i , 0.02* peri, True)


            if area > max_area and len(approx) ==4:
                biggest = approx
                max_area = area

    return biggest ,max_area

def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4,1,2),dtype = np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis =1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

def splitcells(img):
    # each image is 50x50
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes


def sudokumodify(name):

    image = cv2.imread(os.path.join(PROJECT_ROOT, "picture/sud1.jpeg"))
    image = cv2.resize(image, (450,450))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (3,3),6) 
    
    threshold_img = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    contour, hierarchy = cv2.findContours(threshold_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    biggest, maxArea = main_outline(contour)
    if biggest.size != 0:
        biggest = reframe(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])

        # we fist get the perspective tranformation matrix
        # transformation, transalation, and projection(if it is not affine that is not paralel)
        matrix = cv2.getPerspectiveTransform(pts1,pts2)  

        # get the image using the matrix for 450x450 image
        imagewrap = cv2.warpPerspective(image,matrix,(450,450))

        imagewrap =cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)
    else:
        return "sudoku not detected"

    sudoku_cell = splitcells(imagewrap)
    
    Cells_croped = []
    for image in sudoku_cell:    
        img = np.array(image)

        #cropping the image
        img = img[14:42,14:42]

        # getting the digits to be white according to dataset used
        img=255-img
        img = np.asarray(img)
        Cells_croped.append(img)
    return Cells_croped


def predict(Cells_croped):
    result = []
    model=tf.keras.models.load_model("main/anoth_model11.h5")
    for image in Cells_croped:
        img = np.asarray(image)
        img = cv2.resize(img, (28,28))
        img=img.reshape(1,28,28,1)
        predictions = model.predict(img)
        classIndex = np.argmax(predictions[0])
        probabilityValue = np.amax(predictions)
        
        if probabilityValue > 0.3:
            result.append(classIndex)
        else:
            result.append(0)
    grid = np.asarray(result).reshape(9,9)
    grid= grid.reshape(81,1)
    answer=""
    for number in grid:
        answer+=str(number)
    return grid

def process_and_predict(name):
    return predict(sudokumodify(name))
    #return 3


def retrain(image,target):
    model=tf.keras.models.load_model("main/anoth_model11.h5")
    Cells_croped=sudokumodify(image)

    predicted=predict(Cells_croped) 
    X_train=[]  
    Y_train=[]
    for image in Cells_croped:
        img = np.asarray(image)
        img = cv2.resize(img, (28,28))
        img=img.reshape(1,28,28,1)
        X_train.append(img)
    for i in range(len(target)):
        Y_train.append(int(target))
    model.fit(X_train,Y_train,epochs=5)
    model.save("main/anoth_model11.h5")