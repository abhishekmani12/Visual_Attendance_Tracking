import pandas as pd
import dlib
import numpy as np
import PIL.Image
from PIL import ImageFile


hog_face_detector = dlib.get_frontal_face_detector() #HOG based face detector

cnn_face_detection_model ="models/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model) #CNN based face detector

predictor_68_point_model ="models/shape_predictor_68_face_landmarks.dat"
pose_predictor = dlib.shape_predictor(predictor_68_point_model) #shape predictor which generates raw encodings

face_encoder_model = "models/dlib_face_recognition_resnet_model_v1.dat"
face_encoder = dlib.face_recognition_model_v1(face_encoder_model) #model which converts raw rencodings to a 128 embedding

ImageFile.LOAD_TRUNCATED_IMAGES = True


def face_box(img, model,upsample=1): #function to select face detection model
  
    if model == "hog":
        return hog_face_detector(img, upsample) #returns bounding box coordinates localized to the face
    else:
        return cnn_face_detector(img, upsample)
    

def face_landmarks(image, model): #function to generate raw encodings of facial landmarks
    
    face_locations = None
    face_locations = face_box(image, model)
    return [pose_predictor(image, face_location) for face_location in face_locations]


def encoding(file, imgf=False, model="hog", mode="train" ,jitter=1): #function to convert raw encodings to 128 embeddings
   
    if not imgf: #File
        im = PIL.Image.open(file)#PIL format
        im = im.convert('RGB')
    else:
        im=file
    
    image=np.array(im) #numpy array
    
    landmarks = face_landmarks(image, model)
    
    emb=[np.array(face_encoder.compute_face_descriptor(image, landmark_set, jitter)) for landmark_set in landmarks] #embeddings
    
    if mode == "train":
        return emb[0]
    elif mode == "test":
        return emb
    elif mode == "df":
        face_id=file.split("/")[1]
        data={ 'value':emb, 'face_id':face_id}
        return pd.DataFrame(data)
