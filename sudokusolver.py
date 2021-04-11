import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.python.keras.api.keras.preprocessing import image

import keras 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K 

    
img=cv.imread("Sudoku\su.jpg")
# cv.imshow("img",img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("grey",gray)

g_blur=cv.GaussianBlur(gray.copy(), (9,9),0 )
# cv.imshow("Blur990",g_blur)

a_threshold=cv.adaptiveThreshold(g_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,9,2)
# cv.imshow("threshold",a_threshold)

inverted=cv.bitwise_not(a_threshold)
# cv.imshow("inverse",inverted)

kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
dilated=cv.dilate(inverted,kernel)
# cv.imshow("dilated",dilated)

contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


all_areas=[]

for cnt in contours: 
    area=cv.contourArea(cnt)
    all_areas.append(area)

sorted_contours=sorted(contours, key=cv.contourArea, reverse=True)
def contour_finder():
    for c in sorted_contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            # Here we are looking for the largest 4 sided contour
            return approx
corners=contour_finder()
def corner_finder():
    corne = [(corner[0][0], corner[0][1]) for corner in corners]
    top_r, top_l, bottom_l, bottom_r = corne[0], corne[1], corne[2], corne[3]
    return top_l, top_r, bottom_r, bottom_l

tl, tr, br, bl=corner_finder()
# Index 0 - top-right
#       1 - top-left
#       2 - bottom-left
#       3 - bottom-right
cv.circle(gray,(tl[0],tl[1]),5,(0,255,0),-1)
cv.circle(gray,(tr[0],tr[1]),5,(0,255,0),-1)
cv.circle(gray,(bl[0],bl[1]),5,(0,255,0),-1)
cv.circle(gray,(br[0],br[1]),5,(0,255,0),-1)

def crop_warp_img(): 
    width_a=np.sqrt((br[0]-bl[0])**2+(br[1]-bl[1])**2)
    width_b=np.sqrt((tr[0]-tl[0])**2+(tr[1]-tl[1])**2)
    width=max(int(width_a),int(width_b))

    height_a=np.sqrt((tr[0]-br[0])**2+(tr[1]-br[1])**2)
    height_b=np.sqrt((tl[0]-bl[0])**2+(tl[1]-bl[1])**2)
    height=max(int(height_a),int(height_b))


    dimensions=np.array([[0,0],[width - 1, 0],[width - 1, height-1],[0,height-1]],dtype="float32")
    #convert to numpy format 
    ordered_corners=[[tr[0],tr[1]],[tl[0],tl[1]],[bl[0],bl[1]],[br[0],br[1]]]
    ordered_corners=np.array(ordered_corners,dtype="float32")
    #calculating the perspective transform matrix and warp
    #the perspective to grab the screen 
    grid_of_img=cv.getPerspectiveTransform(ordered_corners, dimensions)
    return cv.warpPerspective(img, grid_of_img, (width, height))
    
img=crop_warp_img()

flipVertical = cv.flip(img, 1)

# cv.imshow("final",flipVertical)

grid = cv.cvtColor(flipVertical, cv.COLOR_BGR2GRAY)
grid=cv.bitwise_not(cv.adaptiveThreshold(grid, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 1))

cv.imshow('grid', grid)
edge_horizontal=np.shape(grid)[0]
edge_vertical=np.shape(grid)[1]
cell_edge_horizontal=edge_horizontal//9
cell_edge_vertical=edge_vertical//9

temp_grid=[]
for i in range(cell_edge_horizontal, edge_horizontal+1, cell_edge_horizontal):
    for j in range(cell_edge_vertical,edge_vertical+1,cell_edge_vertical):
        rows=grid[i-cell_edge_horizontal:i]
        temp_grid.append([rows[k][j-cell_edge_vertical:j] for k in range(len(rows))])
#Creating the 9X9 grid of images
final_grid=[]
for i in range(9,len(temp_grid)-8,9):
    final_grid.append(temp_grid[i:i+9])


#MNIST data is split between train and test sets 
(x_train, y_train), (x_test, y_test)=mnist.load_data()  

#Reshape to be a samplpe*pixels*width*height
x_train=np.Reshape(x_train.shape[0], 28 , 28, 1).astype('float32') 
x_test=np.Reshape(x_test.shape[0],28,28,1).astype('float32')

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]

#convert from integers to floats 
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

#normalize to range[0,1]
x_train=(x_train / 255.0 )
x_test=( x_test / 255.0 )

#model summary
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

#compiling model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


cv.waitKey(0)