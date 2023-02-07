import pandas as pd
import dlib
import numpy as np
import PIL.Image
from PIL import ImageFile


hog_face_detector = dlib.get_frontal_face_detector()

cnn_face_detection_model ="models/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

predictor_68_point_model ="models/shape_predictor_68_face_landmarks.dat"
pose_predictor = dlib.shape_predictor(predictor_68_point_model)

face_encoder_model = "models/dlib_face_recognition_resnet_model_v1.dat"
face_encoder = dlib.face_recognition_model_v1(face_encoder_model)

ImageFile.LOAD_TRUNCATED_IMAGES = True
'''
def _css_to_rect(css):
  
    return dlib.rectangle(css[3], css[0], css[1], css[2])
'''


def face_box(img, model,upsample=1):
  
    if model == "hog":
        return hog_face_detector(img, upsample)
    else:
        return cnn_face_detector(img, upsample)
    

def face_landmarks(image, model):
    
    face_locations = None
    face_locations = face_box(image, model)
    
    #face_locations = [_css_to_rect(face_location) for face_location in face_locations]


    return [pose_predictor(image, face_location) for face_location in face_locations]


def encoding(file, model="hog", mode="val" ,jitter=1):
  
    im = PIL.Image.open(file)
    im = im.convert('RGB')
    image=np.array(im)
    
    landmarks = face_landmarks(image, model)
    
    emb=[np.array(face_encoder.compute_face_descriptor(image, landmark_set, jitter)) for landmark_set in landmarks]
    
    if mode == "val":
        return emb[0]
    elif mode == "df":
        face_id=file.split("/")[1]
        data={ 'value':emb, 'face_id':face_id}
        return pd.DataFrame(data)