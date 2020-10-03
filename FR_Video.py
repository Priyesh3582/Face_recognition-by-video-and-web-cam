# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:46:08 2020

@author: PRIYESH
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:29:13 2020

@author: PRIYESH
"""

import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/Users/PRIYESH/Anaconda3/Library/haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
#cap = cv2.VideoCapture('C:/Users/PRIYESH/Videos/Kabira (part1-2) - Yeh Jawaani Hai Deewani - Ranbir Kapoor - Deepika -1080p HD - YouTube (1080p) - Copy.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()




import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/Users/PRIYESH/Anaconda3/Library/haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
#To use a video file as input 
cap = cv2.VideoCapture('C:/Users/PRIYESH/Videos/Kabira (part1-2) - Yeh Jawaani Hai Deewani - Ranbir Kapoor - Deepika -1080p HD - YouTube (1080p) - Copy.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()





#using face_recognition module & opencv
import cv2
import face_recognition

# Get a reference to webcam 
video_capture = cv2.VideoCapture("C:/Users/PRIYESH/Videos/Kabira (part1-2) - Yeh Jawaani Hai Deewani - Ranbir Kapoor - Deepika -1080p HD - YouTube (1080p) - Copy.mp4")

# Initialize variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # Display the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()



