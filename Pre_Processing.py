import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('dd.jpeg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Reading the image as 'img' and converting it to grayscale.

med_val = np.median(img) 
lower = int(max(0, 0.7* med_val))
upper = int(min(255,1.3 * med_val))
edges3 = cv2.Canny(img, 80 ,170)
plt.imshow(edges3,cmap='gray')

# Initial Canny edge detection on the image without any blurring.

blurred_img = cv2.blur(img,(7,7))
plt.imshow(blurred_img,cmap='gray')
edges1 = cv2.Canny(blurred_img,70 ,180)
plt.imshow(edges1,cmap='gray')

# Canny edge detection applied after blurring the image.

BlurIR = cv2.blur(img,(5,5))
retval, threshIR = cv2.threshold(BlurIR,150, 255, cv2.THRESH_BINARY)
plt.imshow(threshIR,cmap='gray')
edgesIR = cv2.Canny(threshIR,50,180)
plt.imshow(edgesIR)

# Canny edge detction applied to a thresholded, blurred image.

contours1,hierarchy1=cv2.findContours(edgesIR,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = np.zeros(threshIR.shape)
for i in range(len(contours1)):
    if hierarchy1[0][i][3] == -1:      
        cv2.drawContours(contours, contours1, i, 255, -1)
plt.imshow(contours,cmap='gray')

# Contour detection and drawing.
