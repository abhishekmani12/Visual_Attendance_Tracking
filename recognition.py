import cv2
import embedding as emb
from sklearn import svm
from PIL import Image
from tqdm import tqdm
import os
import pickle as pkl
import joblib



def train(algorithm):
    
    #loading embedding data
    encodings=joblib.load('Embeddings/encodings.sav')
    face_id=joblib.load('Embeddings/face_ids.sav')
    
    if algorithm == "svm": #setting the required kernel for svm
        
        clf1 = svm.SVC(C=7766.325241554844, kernel="rbf", gamma='auto') #instantiate svm
        clf1.fit(encodings,face_id) #train
        
        pkl.dump(clf1,open('models/svm_classifier.pkl','wb')) #save
        print("SVM Model pickled")
        
        return clf1 #return model object
        
    elif algorithm == "knn":
        
        n=int(round(math.sqrt(len(encodings)))) #get number of neighbors - param
        knn=neighbors.KNeighborsClassifier(n_neighbors=n, weights='distance')
        knn.fit(encodings, face_id)
        
        pkl.dump(knn,open('models/knn_classifier.pkl','wb')) #save
        print("KNN Model pickled")
        
        return knn #return model object




def fitter(algorithm="svm", single_path=None): #Get embeddings for the whole directory or for a single folder - single_path='Dataset/face_name'
       
    encodings=[]
    face_id=[]
    
    emb_path="Embeddings/encodings.sav"
    id_path="Embeddings/face_ids.sav"
    
    if single_path is None:

        master=os.listdir('Dataset/') #get dataset directory


        for face in tqdm(master): #loop through subfolders in Dataset folder
            
            sub=os.listdir("Dataset/" +face)
            
            print(f"{face}'s face encoding in progress")
            
            for faceimg in sub: #loop through images present in each sub folder
                file="Dataset/" + face + "/" + faceimg
                vals=emb.encoding(file,"hog") #pass file to encodings function of embedding script to get 128 face encoding values for each face
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
            vals=emb.encoding(file,"hog") 
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

def pred(file, model): #prediction function - pass image file and required model parameter
    
    result=[]
    vals=emb.encoding(file,"hog") #get encoding
    
    for value in vals:
        res=model.predict([value]) #predict using encoding
        result.append(res[0]) 
    
    return result #return 0th index of the list of list result

def load_model(algorithm="svm"): #Load trained model from pkl format
    
    if algorithm=="svm":
        model=pkl.load(open('models/classifier.pkl', 'rb')) #svm
        
    elif algorithm=="knn":
        model=pkl.load(open('models/knn_classifier.pkl', 'rb')) #knn
    return model