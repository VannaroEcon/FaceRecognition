# Face Recognition with OpenCV and face-recognition
This repository contains the script to detect the face with OpenCV and face-recognition.
The importance point of this project is the fact that we detect and get only good faces (eyes open
and not blurred).

## Libraries
- opencv-python: 4.5.5.62
- face-recognition: 1.3.0
- dlib: 19.23.0

## Method
We first use a built-in model from face-recognition library to detect whether the image
contains faces. Then, we apply a built-in facial landmarks "shape_predictor_68_face_landmarks.dat", to identifies 
locations of eye-brows, eyes, nose, mouth and jawline.

## Usage
1. Run `face_detector.py` to detects if the given image contains a face in it and crops images to get only face of each person.
Cropped face images are stored in 'faces_detected' folder.
2. Run `good_face_selector.py` to chooses the face images that are not closed eyes and too blurred.
Final face images with faces are stored in 'faces_selected' folder.
