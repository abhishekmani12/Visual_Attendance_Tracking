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

* [take_face.py](https://github.com/abhishekmani12/Visual_Attendance_Tracking/blob/main/take_face.py) - Responsible for hands-free automatic dataset creation for a new face based on face angle, mimics Face ID.

* [record.py](https://github.com/abhishekmani12/Visual_Attendance_Tracking/blob/main/record.py) - Temporary script for df manipulation, this serves as the DB for the time being
* [app.py](https://github.com/abhishekmani12/Visual_Attendance_Tracking/blob/main/app.py) - Main Streamlit script which integrates all the other scripts for a demo

## Usecase:
 The system can track attendance on a single face or on a group of faces in a single frame.  
 It requires 5 images of a person's face in various angles for effective training.
 
 - Single face training and batch face trainings are available. A total retrain of the entire dataset can be done if needed.
 - Records will be created automatically for each face as it is added to the dataset.
 - Records will be updated automatically when the attendance for a face is marked.
 - For every new face that is added, the model is trained is ready to take attendance for that face.
   
## Samples:

### Attendance Taking:
![attendance_taking](https://github.com/abhishekmani12/Visual_Attendance_Tracking/assets/76105443/f10a951e-bcdb-4d5c-80f0-552e170ba8e7)

### Uploading images of a face manually:
![Uploading images of a face manually](https://github.com/abhishekmani12/Visual_Attendance_Tracking/assets/76105443/0ec070b0-06e6-4f47-9c33-6828f5159cba)

### Uploading Face Automatically (Face ID mimic):
![Uploading Face Automatically (Face ID mimic)](https://github.com/abhishekmani12/Visual_Attendance_Tracking/assets/76105443/21ed08ad-92a3-4e6b-9655-589c20794c58)

### Records:
![Records](https://github.com/abhishekmani12/Visual_Attendance_Tracking/assets/76105443/7a55ea9b-1699-43a9-96d8-d20dfc411a34)

### Verbose Output - Training on new faces:
![Verbose Output - Training on new faces](https://github.com/abhishekmani12/Visual_Attendance_Tracking/assets/76105443/642e7454-3f71-4629-850b-931ae7a55a0f)



## Pending:
  Containerize app
