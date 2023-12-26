#Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
predictor_file_path = os.path.join(current_directory, "shape_predictor_68_face_landmarks.dat")
#Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
if os.path.isfile(predictor_file_path):
    # File exists, you can now use predictor_file_path in your dlib code
    predictor = dlib.shape_predictor(predictor_file_path)
else:
    print("File not found:", predictor_file_path)

#status marking for current state
sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

def compute(ptA,ptB):
	dist = np.linalg.norm(ptA - ptB)
	return dist

def blinked(a,b,c,d,e,f):
	up = compute(b,d) + compute(c,e)
	down = compute(a,f)
	ratio = up/(2.0*down)

	#Checking if it is blinked
	if(ratio>0.22):
		return 2
	elif(ratio>0.18 and ratio<=0.22):
		return 1
	else:
		return 0

def lip_distance(landmarks):
    top_lip = landmarks[50:53]
    top_lip = np.concatenate((top_lip, landmarks[61:64]))

    low_lip = landmarks[56:59]
    low_lip = np.concatenate((low_lip, landmarks[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])

    global yawning

    if distance > 20:
            yawning += 1
    else:
            yawning = 0

    if yawning > 30:
        return 1
    else:
        return 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    #detected face in faces array

    face_frame = None
    
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        #The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36],landmarks[37], 
        	landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
        	landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        lip=lip_distance(landmarks)
        
        #Now judge what to do for the eye blinks
        if(left_blink==0 or right_blink==0):
        	sleep+=1
        	drowsy=0
        	active=0
        	if(sleep>6):
        		status="SLEEPING !!!"
        		color = (255,0,0)

        elif(left_blink==1 or right_blink==1 or lip==1):
        	sleep=0
        	active=0
        	drowsy+=1
        	if(drowsy>6):
        		status="Drowsy !"
        		color = (0,0,255)

        else:
        	drowsy=0
        	sleep=0
        	active+=1
        	if(active>6):
        		status="Active :)"
        		color = (0,255,0)
        	
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

        for n in range(0, 68):
        	(x,y) = landmarks[n]
        	cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    if face_frame is not None:
        cv2.imshow("Result of detector", face_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
