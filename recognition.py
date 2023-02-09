import cv2
import embedding as emb
from sklearn import svm
from PIL import Image
from tqdm import tqdm
import os
import pickle as pkl

def train(algorithm="svm"): #classifier train function
    encodings = []
    face_id = []


    master=os.listdir('Dataset/') #get dataset directory


    for face in tqdm(master): #loop through subfolders in Dataset folder
        sub=os.listdir("Dataset/" +face)
        print(face,"'s face training in progress")
        for faceimg in sub: #loop throuigh images present in each sub folder
            file="Dataset/" + face + "/" + faceimg
            vals=emb.encoding(file,"hog") #pass file to encodings function of embedding script to get 128 face encoding values for each face
            encodings.append(vals) #appending encodings to list
            face_id.append(face) #append face name to list

    if algorithm == "svm":
        clf1 = svm.SVC(gamma='scale') #instantiate svm
        clf1.fit(encodings,face_id) #train
        pkl.dump(clf1,open('models/classifier.pkl','wb')) #save
        print("Trained Model pickled")
        return clf1 #return model object

        
    
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
    vals=emb.encoding(file,"hog") #get encoding
    res=model.predict([vals]) #predict using encoding
    
    return res[0] #return 0th index of the list of list result

def loadmodel(): #Load trained model from pkl format
    model = model=pkl.load(open('models/classifier.pkl', 'rb'))
    return model