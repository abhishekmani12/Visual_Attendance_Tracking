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



st.title("Visual Attendance Tracker")
st.sidebar.title('Configure')


source = ("Take Live Attendance", "Take Photo Attendance", "Upload New Face","Take New Face", "Retrain", "Get Attendance")
source_index = st.sidebar.selectbox("Input", range(len(source)), format_func=lambda x: source[x])
st.sidebar.write("####")

models=("SVM", "KNN")
model_option=st.sidebar.selectbox("Model", range(len(models)), format_fun= lambda x: source[x])

model=rec.load_model(algorithm=model_option.lower())

st.sidebar.write("####")


if source_index == 0 :
    
    fc=0
    Name=""
    queue=[]*2
    queue.insert(0,None)
    model_option="SVM"
    model=rec.load_model(algorithm=model_option.lower())
    temp=""
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        fc+=1

        if fc in range(1, 10):
            continue

        _, img=cap.read()

        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        queue.insert(1,Name)
        Name=rec.pred(img, model, imgf=True)


        if len(Name) == 1:
            if not temp == Name[0]:
                st.write(Name[0])
                result=r.update(Name[0])
                st.write(result)

            temp=Name[0]
        
elif source_index == 1:
    st.write("in progress")
                     
                      
elif source_index == 2:
    
    face_id=st.text_input("Enter your name")
    uploaded_files = st.file_uploader("Upload 5 images", accept_multiple_files=True)

    Directory="Dataset/"+face_id

    if face_id:
        if os.path.exists(Directory):
            st.write("Face ID already Exists! Please reselect this option")
            exit()
        else:
            os.makedirs(Directory)
            st.write("New Face ID!")

        if uploaded_files:
            for File in uploaded_files:
                save_path=Path(Directory, File.name)

                with open(save_path, mode='wb') as w:
                    w.write(File.getvalue())

        rec.fitter(algorithm=model_option.lower(), take_face_live=False, name=None, single_path=Directory)        
       
                      
elif source_index == 3:
        
        face_id=st.text_input("Enter your name")

            Directory="Dataset/"+face_id

            if face_id:
                if os.path.exists(Directory):
                    st.write("Face ID already Exists! Please reselect this option")
                    exit()
                else:
                    rec.fitter(algorithm=model_option.lower(), take_face_live=True, name=face_id, single_path=None)
                    st.write("Training New Face done")

elif source_index == 4:
        
    #entire dataset gets retrained
        if st.button("Train on the whole Dataset"):               
            fitter(algorithm="svm", take_face_live=False, name=None, single_path=None)
            st.write("Training Done")
                 

elif source_index == 5:
        
        df=r.fetch()
        st.dataframe(df)
                      
        if st.button("Save current record to a new file"):
            output=r.save()
            st.write(output)
