from django.shortcuts import render
import numpy as np # linear algebra
import cv2
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
    for i in contour:
        area = cv2.contourArea(i)
        if area >50:
            peri = cv2.arcLength(i, True)
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
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes
def sudokumodify(request):
    image = cv2.imread('main/sud1.jpeg')
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
        matrix = cv2.getPerspectiveTransform(pts1,pts2)  
        imagewrap = cv2.warpPerspective(image,matrix,(450,450))
        imagewrap =cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)
    
    sudoku_cell = splitcells(imagewrap)
    
    Cells_croped = []
    for image in sudoku_cell:
        
        img = np.array(image)
        img = img[14:42,14:42]
        img=255-img
        img = np.asarray(img)
        Cells_croped.append(img)
            
        
    result = []
    model=tf.keras.models.load_model("main/anoth_model11.h5")
    for image in Cells_croped:
        img = np.asarray(image)
        img = cv2.resize(img, (28,28))
        img=img.reshape(1,28,28,1)
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)
        
        if probabilityValue > 0.3:
            result.append(classIndex[0])
        else:
            result.append(0)
    grid = np.asarray(result).reshape(9,9)
    return grid
def predict(request):
    model=tf.keras.models.load_model("main/anoth_model11.h5")
    result=0
    imgData = cv2.imread('main/img1.png')
    imgData = cv2.resize(imgData, (28,28))
    
    imgData =cv2.cvtColor(imgData, cv2.COLOR_BGR2GRAY)
    for i in range(28):
        for j in range(28):
            if imgData[i][j]>200:
                imgData[i][j]=255
    img = np.asarray(imgData)
    result=img.shape
    
    img=img.reshape(1,28,28,1)
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    
    if probabilityValue > 0.3:
        result=(classIndex[0])
    else:
        result=(0)
    return (result)
def index(request):
    digit=sudokumodify(request)
    return render(request, 'index.html', {'digit':digit})
def fir(request):
    return render(request,"index.html",{'digit':3})
# urls.py
from django.urls import path
from .views import index
urlpatterns = [
    path('main', index, name="index"),
    path('', fir, name="fir")
]