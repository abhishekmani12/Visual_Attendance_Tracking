import mediapipe as mp
import cv2
import numpy as np
import os
import time


def take_photo(face_id):
    count=0
    fc=1
    track=None
    current=""
    
    if face_id is None:
        print("Please enter your face id")
        return None
    
    face_path="Dataset/"+face_id
    
    if not os.path.exists(face_path):
        os.makedirs(face_path)
    else:
        print("Face already exists!")
        return "Exists"

    baseobj=mp.solutions.face_mesh
    base_model=baseobj.FaceMesh(min_detection_confidence=0.4, min_tracking_confidence=0.4)

    track="focused"
    cap =cv2.VideoCapture(0)
    
    msg=""
    
    while cap.isOpened():
        fc+=1
        
        s, img=cap.read()
        
        if fc in range(1,100) or fc%2 == 0:
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
                
        cv2.imshow('focus', img)
        
        if count == 5:
            print("Process finished")
            break
            
        
        if cv2.waitKey(1) & 0xFF == ord('f'):
            print("Process ended by escape key")
            break

    cap.release()
    cv2.destroyAllWindows()
    return face_path
