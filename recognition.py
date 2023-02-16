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
        print(f"{face}'s face encoding in progress")
        for faceimg in sub: #loop through images present in each sub folder
            file="Dataset/" + face + "/" + faceimg
            vals=emb.encoding(file,"hog") #pass file to encodings function of embedding script to get 128 face encoding values for each face
            encodings.append(vals) #appending encodings to list
            face_id.append(face) #append face name to list
    
    if algorithm == "svm": #setting the required kernel for svm
        
        clf1 = svm.SVC(C=7766.325241554844, kernel="rbf", gamma='auto') #instantiate svm
        clf1.fit(encodings,face_id) #train
        
        pkl.dump(clf1,open('models/svm_classifier.pkl','wb')) #save
        print("SVM Model pickled")
        return clf1 #return model object
        
    elif algorithm == "knn":
        
        n=int(round(math.sqrt(len(encodings))))
        knn=neighbors.KNeighborsClassifier(n_neighbors=n, weights='distance')
        knn.fit(encodings, face_id)
        
        pkl.dump(knn,open('models/knn_classifier.pkl','wb')) #save
        print("KNN Model pickled")
        return knn #return model object
        
    '''
    res=sklearn.model_selection.cross_val_score(clf1, encodings, face_id, n_jobs=-1, cv=3)
    accuracy=res.mean()
    print(accuracy)
    '''
        
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

def load_model(algorithm="svm"): #Load trained model from pkl format
    
    if algorithm=="svm":
        model=pkl.load(open('models/classifier.pkl', 'rb')) #svm
        
    elif algorithm=="knn":
        model=pkl.load(open('models/knn_classifier.pkl', 'rb')) #knn
    return model