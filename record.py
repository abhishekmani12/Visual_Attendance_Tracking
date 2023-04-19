import pandas as pd
import os 
from datetime import datetime


Template_df=None
curr_df=None

def update_global():
    
    global Template_df
    Template_df=pd.read_csv("attendance.csv")
    global curr_df
    curr_df=Template_df.copy()
    
def set_new():

    id_path="Embeddings/face_ids.sav"
    face_id=pd.Series(joblib.load(id_path)).unique()
    df=pd.DataFrame(face_id,columns=['Face_id'])
    df.insert(1,'Status','absent')
    df.to_csv("attendance.csv",index=False)
    update_global()

update_global()


def fetch():
    
    return curr_df


def update(name):
    
    if curr_df.loc[curr_df.Face_id == name, "Status"].item() == "Present":
        return "Already present"
    else:
        curr_df.loc[curr_df.Face_id == name, "Status"]="Present"
        return "Marked Present"
    
def save():

    now = datetime.now()
    dt = now.strftime("%d-%m-%Y__%H:%M:%S")
    
    loc="record_folder/attendance"+"_"+dt+".csv"
    
    curr_df.to_csv(loc, index=False)
    
    return "Attendance Record Saved"

def add(name):
    
    if (Template_df.Face_id == name).any():
        print("Face ID Exists")
    else:
        Template_df.loc[len(Template_df.index)] = [name,'absent']
        Template_df.to_csv("attendance.csv", index=False)
        update_global()
