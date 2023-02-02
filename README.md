# Visual Attendance Tracking
Deep Learning based attendance tracker using visual capture

## Components:
1. DNN - Face Tracker
2. Dlib - Encoding Face with 26 landmarks to perform shearing and other relative image transformations.
3. OpenFace - Model to predict Face Landmarks to generate 128 Vector Embeddings of a face.
4. LightGBM/XGBoost Reg/SVM - To predict Face_id when given an embedding (Face_id = Register Number).
* Lookup Table to reference names
* Front end for image capture and portal to save attendance records

