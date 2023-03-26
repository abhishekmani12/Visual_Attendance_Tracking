import pandas as pd
import os 
from datetime import datetime



import pandas as pd
Template_df=pd.read_csv("attendance.csv")
curr_df=Template_df.copy()

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
