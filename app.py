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
import mediapipe as mp
import numpy as np
import time

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
                    print(Name[0])
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
    
    wh=st.empty()
    
    Directory="Dataset/"+face_id

    if face_id:
        if os.path.exists(Directory):
            with wh.container():
                st.warning("WARNING: Face ID Already exists!")

        os.makedirs(Directory,exist_ok = True)

        if st.button("Register"):
            wh.empty()
            for File in uploaded_files:
                save_path=Path(Directory, File.name)

                with open(save_path, mode='wb') as w:
                    w.write(File.getvalue())
            
            with st.spinner("Training in Progress"):
                rec.fitter(algorithm=models[model_option].lower(), take_face_live=False, name=None, single_path=Directory)
                r.add(face_id)
            with wh.container():
                st.success("Training Done", icon="✅")       
       
                      
elif source_index == 3:

    face_id=st.text_input("Enter your name")   

    frameST = st.image([])
    cap = cv2.VideoCapture(0)


    wh=st.empty()
    count=0
    fc=1
    track=None
    current=""


    face_path="Dataset/"+face_id

    if face_id:
        if os.path.exists(face_path):
            with wh.container():
                st.warning("WARNING: Face ID Already exists!")

        os.makedirs(face_path,exist_ok = True)


    baseobj=mp.solutions.face_mesh
    base_model=baseobj.FaceMesh(min_detection_confidence=0.4, min_tracking_confidence=0.4)

    track="focused"

    msg=""
    flag=st.button("Take Face")

    while flag:
            wh.empty()
            fc+=1

            s, img=cap.read()

            if fc in range(1,50) or fc%2 == 0:
                continue


            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #fm requires rgb input
            landmarks=base_model.process(img) #get keypoint landmarks mesh for a face
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

            #Extracting eye edge(left - 33, right - 263), nose(1), mouth edge(left - 61, right - 291), chin(199) keypoint landmarks

            twoD=[] #x,y
            threeD=[] #axis

            h, w, c=img.shape

            if landmarks.multi_face_landmarks:
                for dat in landmarks.multi_face_landmarks:
                    for i, cood in enumerate(dat.landmark):
                            if i == 33 or i == 263 or i == 1 or i == 61 or i == 291 or i == 199:

                                x=int(cood.x*w) #multiplying width to x cood and height to y cood
                                y=int(cood.y*h)
                                z=cood.z


                                twoD.append([x,y])


                                threeD.append([x,y,z])


                    twoD=np.array(twoD, dtype=np.float64)
                    threeD=np.array(threeD, dtype=np.float64)


                    focalpoint=1*w #fx, fy
                    skew=0 #gamma
                    u_cood=h/2
                    v_cood=w/2

                    #camera matrix

                    cam_mat=np.array([
                                        [focalpoint, 0, u_cood],
                                        [0, focalpoint, v_cood],
                                        [0, 0, 1]
                                    ])

                    #distance matrix
                    dist_mat=np.zeros((4,1), dtype=np.float64)

                    #pnp - convert 3d point in obj cood frame to 2d camera cood frame by getting rotation and translation vectors
                    s, rot_v, trans_v=cv2.solvePnP(threeD, twoD, cam_mat, dist_mat)

                    rot_mat, _ = cv2.Rodrigues(rot_v) # convert to matrix to get rot angle

                    angle, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_mat) #extract angles

                    xdegree=angle[0]*360
                    ydegree=angle[1]*360

                    lp=face_path+"/"+face_id+"-left.png"
                    rp=face_path+"/"+face_id+"-right.png"
                    bp=face_path+"/"+face_id+"-bottom.png"
                    tp=face_path+"/"+face_id+"-top.png"
                    cp=face_path+"/"+face_id+"-centre.png"

                    trash_path=face_path+"/"+face_id+"-trash.png"


                    if ydegree < -10:  #RIGHT
                        current="right"
                        if not os.path.exists(rp) and track == "right":
                            cv2.imwrite(rp,img)
                            msg="Tilt your face to the LEFT"
                            print(msg)
                            count+=1

                    elif ydegree > 10: #LEFT
                        current="left"
                        if not os.path.exists(lp) and track == "left":
                            cv2.imwrite(lp,img)
                            msg="All photos taken"
                            print(msg)
                            count+=1

                    elif xdegree < -4: #BOTTOM
                        current="bottom"
                        if not os.path.exists(bp) and track == "bottom":
                            cv2.imwrite(bp,img)
                            msg="Tilt your face to the RIGHT"
                            print(msg)
                            count+=1

                    elif xdegree > 5: # TOP
                        current="top"
                        if not os.path.exists(tp) and track == "top":
                            cv2.imwrite(tp,img)
                            msg="Tilt your face down"
                            print(msg)
                            count+=1

                    else:             #CENTRE
                        current="centre"
                        if not os.path.exists(cp) and track == "centre":
                            cv2.imwrite(cp,img)
                            msg="Tilt your face UP"
                            print(msg)
                            count+=1

                    track=current       

            cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, 4)
                
            frameST.image(img, channels="BGR")

            if count == 5:
                rec.fitter(algorithm=models[model_option].lower(), take_face_live=False, name=None, single_path=face_path)
                r.add(face_id)
                print("Process finished")
                frameST.empty()
                
                with wh.container():
                    st.success("Face Has been registered", icon="✅")
                flag=False
                
            
    else:
        cap.release()

                
elif source_index == 4:
        
    #entire dataset gets retrained
        if st.button("Train on the whole Dataset"):
            
            with st.spinner("Training in Progress"):
                rec.fitter(algorithm=models[model_option].lower(), take_face_live=False, name=None, single_path=None)
                r.set_new()
            st.success("Training Done", icon="✅")
                 

elif source_index == 5:
        
        df=r.fetch()
        st.table(df)
                      
        if st.button("Save current record to a new file"):
            output=r.save()
            st.success(output, icon="✅")
