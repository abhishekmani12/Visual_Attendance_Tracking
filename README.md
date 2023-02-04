# Visual Attendance Tracking
Deep Learning based attendance tracker using visual capture

## Components:
1. DNN - Face Tracker
2. Dlib - Encoding Face with 26 landmarks to perform shearing and other relative image transformations.
3. OpenFace - Model to predict Face Landmarks to generate 128 Vector Embeddings of a face.
4. LightGBM/XGBoost /SVM - To predict Face_id when given an embedding (Face_id = Register Number).
* Lookup Table to reference names
* Front end for image capture and portal to save attendance records

## Workflow:

### TRAINING:

1. Store atleast 5 images of each person in each folder in the same directory labelled with the name.
2. Pass these folders to pose detection, allignment and crop scripts. This creates a new directory which contains the transformed images.
3. Pass the new alligned directory to the OpenFace model to generate the 128 embeddings for each image in the form of a CSV file in the same directory.
4. Pass the CSV file to a Classifier to train [ Features: 128 embedding | Target: face_id ] with label encoding to assist in training.


### INFERENCE:

1. Take a picture from a video stream with OpenCV.
2. Identify Face in the picture and crop it to the face and pass to the next step with OpenCV and Dlib.
3. Pass the Face Image to pose detection and alignment script to be transformed accordingly.
4. Pass the transformed image to OpenFace model to generate 128 embedding csv.
5. Pass CSV to the trained classifier model to classify which face the embedding belongs to.

DEPLOYMENT: Initial test deployment through streamlit for feasability.

