# Visual Attendance Tracking
Deep Learning based attendance tracker using visual capture

## Components:
1. DNN/"hog" - Face Tracker
2. Dlib - Encoding Face with 26 landmarks to perform shearing and other relative image transformations.
3. Dlib_Shape_Predictor - Model to predict Face Landmarks to generate 128 Vector Embeddings of a face.
4. KNN/SVM - To predict Face_id when given an embedding (Face_id = Register Number).
* Front end for image capture and portal to save attendance records

## Scripts:
* [embedding.py](https://github.com/abhishekmani12/Visual_Attendance_Tracking/blob/main/embedding.py) - Generates the face embeddings. Contains computation  for face detector, face landmark identifier and embeddings generator models.  

* [recognition.py](https://github.com/abhishekmani12/Visual_Attendance_Tracking/blob/main/recognition.py) - Has Training, Prediction and Compression functions. Imports embeddings.py. Contains Computation for SVM/KNN classifier models.  

* [take_face.py](https://github.com/abhishekmani12/Visual_Attendance_Tracking/blob/main/take_face.py) - Responsible for hands-free automatic dataset creation for a new face based on face angle.

* [record.py](https://github.com/abhishekmani12/Visual_Attendance_Tracking/blob/main/record.py) - Temporary script for df manipulation, this serves as the DB for the time being
* [app.py](https://github.com/abhishekmani12/Visual_Attendance_Tracking/blob/main/app.py) - Main Streamlit script which integrates all the other scripts for a demo

## Usecase:
 The system can track attendance on a single face or on a group of faces in a single frame.  
 It requires 5 images of a person's face in various angles for effective training.

## Pending:
  Updating readme with photos and videos  
  Containerize app
