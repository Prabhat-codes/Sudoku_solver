import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#read image
def read_img():
    print("Enter image name: ")
    image_url = input()
    img = cv.imread(image_url,cv.IMREAD_GRAYSCALE)
    return img
def show_img(img):
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
# blurring(reduces noise), thresholding(black & white), dilating(focusing on main image)
def process(img):
    #blur = cv.GaussianBlur(img,(5,5),1)
    thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,10,3)
    inv = cv.bitwise_not(thresh)
    kernel = np.ones((1,1),np.uint8)
    dial = cv.dilate(inv,kernel,iterations=1)
    return dial

#finding the biggest contour i.e sudoku
def biggestCont(contours,perc=0.05):
    biggest = np.array([])
    max_peri = 0
    for c in contours:
        peri = cv.arcLength(c,True)
        approx = cv.approxPolyDP(c,perc*peri,True)
        if peri>max_peri and approx.shape[0]==4:
            biggest = approx
            max_peri = peri
    return biggest 

def reorder(pts):
    pts = np.reshape(pts,(4,2))
    newPt = np.zeros((4,1,2))
    newPt[0] = pts[np.argmin(pts.sum(axis=1))]
    newPt[3] = pts[np.argmax(pts.sum(axis=1))]
    newPt[1] = pts[np.argmin(np.diff(pts,axis=1))]
    newPt[2] = pts[np.argmax(np.diff(pts,axis=1))]
    return newPt  


def removelines(new,t=3):	#try not to use. doesn't generalize
    cann = (cv.Canny(new,80,180,0))
    lines = cv.HoughLines(cann,1,np.pi/180,200)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(new,(x1,y1),(x2,y2),0,t+3)
    return new  

def split(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes  

def cropbox(boxes):
    a = np.zeros((81,28,28,1)) #81 boxes of size 28*28
    for i in range(81):
        img = np.array(boxes[i])
        #img = cv.erode(img,np.ones((1,1),np.uint8)) #erode
        img = img[4:img.shape[0]-3,3:img.shape[1]-3] #cropping noisy edges
        img = cv.resize(img,(28,28))
        a[i] = img.reshape((28,28,1))
    return a 

def dispNo(no,img = np.zeros((450,450,3)),color=(255,0,255),fs=5):
    h = img.shape[0]//9
    w = img.shape[1]//9
    for i in range(9):
        for j in range(9):
            if no[i*9 + j]!=0:
                cv.putText(img,str(no[i*9 + j]),(int(j*w + 0.3*w),int(i*h + 0.7*h)),cv.FONT_HERSHEY_COMPLEX_SMALL,fs,color,1,cv.LINE_AA)
    return img

model = tf.keras.models.load_model('digit.h5')
import import_ipynb
import sudokuSolver

width = 450
height = 450

#main
image = read_img()
img = process(image)
contours,hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
biggest = biggestCont(contours)
biggest = reorder(biggest)
pts1 = np.float32(np.reshape(biggest,(4,2)))
pts2 = np.float32([[0,0],[width-1,0],[0,height-1],[width-1,height-1]])
M = cv.getPerspectiveTransform(pts1,pts2)
new = cv.warpPerspective(img,M,(width,height))
new = removelines(new,5)
#progress = []
#progress.append(np.array(cv.resize(cv.cvtColor(image,cv.COLOR_GRAY2BGR),(450,450)),np.uint8)) 

boxes = split(new)
#a = cropbox(boxes) #### REMEMBER TO NORMALIZE "a" ####
a = tf.keras.utils.normalize(a, axis=1)  
#plt.imshow(a[25],cmap=plt.cm.binary_r)

pred = model.predict(a)
pred = pred.argmax(axis=1) * (np.amax(pred,axis=1)>0.8)
progress.append(np.array(dispNo(pred,np.zeros((450,450)),color=(255,0,255)),np.uint8))

pos = (pred==0)*np.ones(81)
pos

board = pred.reshape((9,9))
try:
    sudokuSolver.solve(board)
except:
    pass
board
