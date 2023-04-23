import cv2
import embedding as emb
from sklearn import svm
from sklearn import neighbors
from PIL import Image
from tqdm import tqdm
import os
import pickle as pkl
import joblib
import math
import take_face as tf



def train(algorithm):
    
    #loading embedding data
    encodings=joblib.load('Embeddings/encodings.sav')
    face_id=joblib.load('Embeddings/face_ids.sav')
    pkl_svm='models/svm_classifier.pkl'
    pkl_knn='models/knn_classifier.pkl'
    
    if algorithm == "svm": #setting the required kernel for svm
        
        clf1 = svm.SVC(C=7766.325241554844, kernel="rbf", gamma='auto') #instantiate svm
        clf1.fit(encodings,face_id) #train
        if os.path.exists(pkl_svm):
            os.remove(pkl_svm)
        pkl.dump(clf1,open(pkl_svm,'wb')) #save
        print("SVM Model pickled")
        
        return clf1 #return model object
        
    elif algorithm == "knn":
        
        n=int(round(math.sqrt(len(encodings)))) #get number of neighbors - param
        knn=neighbors.KNeighborsClassifier(n_neighbors=n, weights='distance')
        knn.fit(encodings, face_id)
        if os.path.exists(pkl_knn):
            os.remove(pkl_knn)
        pkl.dump(knn,open(pkl_knn,'wb')) #save
        print("KNN Model pickled")
        
        return knn #return model object




def fitter(algorithm="svm", take_face_live=False, name=None, single_path=None): #Get embeddings for the whole directory or for a single folder - single_path='Dataset/face_name'
       
    encodings=[]
    face_id=[]
    
    emb_path="Embeddings/encodings.sav"
    id_path="Embeddings/face_ids.sav"
    
    if(take_face_live): #for training a new face with a live camera
        
        print("Ensure that your face is level with the camera and tilt your head slowly")
        if name is None:
            name=input("Enter your name: ")
        single_path=tf.take_photo(name) 
        
    if single_path == "Exists":
        return None
    
    elif single_path is None:

        master=os.listdir('Dataset/') #get dataset directory


        for face in tqdm(master): #loop through subfolders in Dataset folder
            
            sub=os.listdir("Dataset/" +face)
            
            print(f"{face}'s face encoding in progress")
            
            for faceimg in sub: #loop through images present in each sub folder
                file="Dataset/" + face + "/" + faceimg
                vals=emb.encoding(file, model="hog", mode="train" ) #pass file to encodings function of embedding script to get 128 face encoding values for each face
                encodings.append(vals) #appending encodings to list
                face_id.append(face) #append face name to list
                
    else:
        
        #loading embedding data
        encodings=joblib.load(emb_path)
        face_id=joblib.load(id_path)
        
        sub=os.listdir(single_path)
        face=single_path.split("/")[1] #getting face name from path
        
        print(f"{face}'s face encoding in progress")
        
        for faceimg in sub: #loop through images present in each sub folder
            file="Dataset/" + face + "/" + faceimg
            vals=emb.encoding(file, model="hog", mode="train") 
            encodings.append(vals) 
            face_id.append(face) 
             
    if os.path.exists(emb_path) and os.path.exists(id_path): #deleting existing embedding files
        os.remove(emb_path)
        os.remove(id_path)
    
    joblib.dump(encodings, emb_path) #saving to file
    joblib.dump(face_id, id_path)
       

    return train(algorithm)
            
        
def compressor(file): #image compressor function
    
    filepath = os.path.join(os.getcwd(), file)

    image = Image.open(filepath)

    image.save(file,"JPEG", 
                 optimize = True, 
                 quality = 50)
    return

def folder_compressor(folder): #batch compressor function
    sub=os.listdir("Dataset/"+folder)
    for faceimg in sub:
        file="Dataset/" + face + "/" + faceimg
        compressor(file)
    
    return 

def pred(file, model, imgf=False): #prediction function - pass image file and required model parameter
    
    result=[]
    vals=emb.encoding(file, imgf, model="hog", mode="test") #get encoding
    
    for value in vals:
        res=model.predict([value]) #predict using encoding
        result.append(res[0]) 
    
    return result #return 0th index of the list of list result

def load_model(algorithm="svm"): #Load trained model from pkl format
    
    model=None
    if algorithm=="svm":
        model=pkl.load(open('models/svm_classifier.pkl', 'rb')) #svm
        
    elif algorithm=="knn":
        model=pkl.load(open('models/knn_classifier.pkl', 'rb')) #knn
    return model


