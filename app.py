import cv2
import pandas as pd
import os
from io import StringIO
from pathlib import Path

import streamlit as st
import ffmpeg

import embedding as emb
import recognition as rec
import record as r
import take_face as tk


import cv2
import pandas as pd
import os
from io import StringIO
from pathlib import Path

import streamlit as st
import ffmpeg

import embedding as emb
import recognition as rec
import record as r
import take_face as tk
        
    
    
model_option=0
model=None
st.title("Visual Attendance Tracker")
st.sidebar.title('Configure')


source = ("Take Live Attendance", "Take Photo Attendance", "Upload New Face","Take New Face", "Retrain", "Get Attendance")

source_index = st.sidebar.selectbox("Input", range(len(source)), format_func=lambda x: source[x])
st.sidebar.write("####")

models=("SVM", "KNN")

if source_index != 5:
    model_option=st.sidebar.selectbox("Model", range(len(models)), format_func= lambda x: models[x])

    model=rec.load_model(algorithm=models[model_option].lower())

st.sidebar.write("####")


if source_index == 0 :
        ph = st.empty()
        start=st.sidebar.checkbox('Start')
        
        frameST = st.image([])
        cap = cv2.VideoCapture(0)
        
        fc=0
        Name=None
        result=None
        queue=[]*2
        queue.insert(0,None)
        model_option="SVM"
        model=rec.load_model(algorithm=model_option.lower())
        temp=""
        
        
        while start:
            
            ret, img = cap.read()

            fc+=1

            if fc in range(1, 10):
                continue


            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            queue.insert(1,Name)
            Name=rec.pred(img, model, imgf=True)


            if len(Name) == 1:
                if not temp == Name[0]:
                    result=r.update(Name[0])
                    ph.empty()
                    with ph.container():
                        st.success(f"Face ID: {Name[0]}  STATUS: {result}",  icon="✅")
                temp=Name[0]
                
            img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
            frameST.image(img, channels="BGR")
            
        else:
            cap.release()
            ph.empty()
            with ph.container():
                st.warning("Attendance Tracking is stopped")
            
        
elif source_index == 1:
    st.info("Work in progress")
                     
                      
elif source_index == 2:
    
    uploaded_files = st.file_uploader("Upload 5 images", accept_multiple_files=True)
    face_id=st.text_input("Enter your name")

    Directory="Dataset/"+face_id

    if face_id:
        if os.path.exists(Directory):
            st.error("Face ID already Exists! Please reselect this option")
            exit()
        else:
            os.makedirs(Directory)
            st.warning("New Face ID!")

        if uploaded_files:
            for File in uploaded_files:
                save_path=Path(Directory, File.name)

                with open(save_path, mode='wb') as w:
                    w.write(File.getvalue())
            
            with st.spinner("Training in Progress"):
                rec.fitter(algorithm=models[model_option].lower(), take_face_live=False, name=None, single_path=Directory)
            st.success("Training Done", icon="✅")       
       
                      
elif source_index == 3:
        
        face_id=st.text_input("Enter your name")

        Directory="Dataset/"+face_id

        if face_id:
            if os.path.exists(Directory):
                st.error("Face ID already Exists! Please reselect this option")
                exit()
            else:
                
                with st.spinner("Training in Progress. Follow instructions specified in the video output"):
                    rec.fitter(algorithm=models[model_option].lower(), take_face_live=True, name=face_id, single_path=None)
                st.success("Training New Face done", icon="✅")

elif source_index == 4:
        
    #entire dataset gets retrained
        if st.button("Train on the whole Dataset"):
            
            with st.spinner("Training in Progress"):
                rec.fitter(algorithm="svm", take_face_live=False, name=None, single_path=None)
                
            st.success("Training Done", icon="✅")
                 

elif source_index == 5:
        
        df=r.fetch()
        st.dataframe(df)
                      
        if st.button("Save current record to a new file"):
            output=r.save()
            st.success(output, icon="✅")
