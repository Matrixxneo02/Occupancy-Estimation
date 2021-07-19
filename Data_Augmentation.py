import cv2
import tensorflow.keras
import matplotlib.pyplot as plt
%matplotlib inline

#These are the inputs which help in the Data Augmentation.

cap= cv2.VideoCapture('Input Video Here',0)
i=1
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%50 == 0:    # The 50 here indicates the frame gap for picking images.
        cv2.imwrite('framenumber'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()

# This is the code for splitting the video sequence at a gap of 50 frames.

from keras.preprocessing.image import ImageDataGenerator
augment = ImageDataGenerator(
    rotation_range=5,
    zoom_range=.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

plt.imshow(augment.random_transform(img)
           
# This creates the augment object which when run on images with
# the random transform, subjects the image to a random set of 
# transformations out ot the inputs provided.
