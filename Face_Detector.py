"""
This is a Face Detector AI Algorithm. It works only on grayscale images. This is because haarcascade algorithm is used in this code, haarcascade works only on grayscale images
"""

import cv2      # pip install opencv-python
from random import randrange

# Load some pre-trained data on frontal face from opencv (https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')        # cv2.CascadeClassifier means opencv will classify faces from the cascade algorithm

# Choose an image to detect face  ( We will convert the image into grayscale later in the code )
img1 = cv2.imread('elon_musk_face.jpg')

# Converting the image into grayscale
grayscaled_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Detect Face
face_coordinates1 = trained_face_data.detectMultiScale(grayscaled_img1)
# print(face_coordinates1)

# Draw rectangles around the faces
# (x, y, w, h) = face_coordinates1[0]
for (x, y, w, h) in face_coordinates1:
    cv2.rectangle(img1, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)  # cv2.rectangle(image, (x, y), (x+w, y+h), (BGR color), thickness of the rectangle)

# Display the image with the faces spotted
cv2.imshow('Face Detector AI', img1)        # This will pop-up a window named "Face Detector AI" with the image. (Make sure to use 'cv2.waitKey()', otherwise the window would close immediately)

# Wait here in the code and listen for a keypress
cv2.waitKey()

# End of the Code
print("Code Completed")
