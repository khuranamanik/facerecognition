import cv2
import numpy as np
import os 
flag_id1=0
flag_id2=0
flag_id3=0
flag_id4=0
#Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
# Create Local Binary Patterns Histograms for face recognization
model = cv2.face.LBPHFaceRecognizer_create()
assure_path_exists("saved_model/")
# Load the  saved pre trained mode
model.read('saved_model/s_model.yml')
# Load prebuilt classifier for Frontal Face detection
# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# font style
font = cv2.FONT_HERSHEY_COMPLEX
# Initialize and start the video frame capture from webcam
cam = cv2.VideoCapture(0)
# Looping starts here
while True:
    # Read the video frame
    ret, im =cam.read()
    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # Getting all faces from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5) #default
    # For each face in faces, we will start predicting using pre trained model
    for(x,y,w,h) in faces:
        # Create rectangle around the face
        #cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 2)
        # Recognize the face belongs to which ID
        Id, confidence = model.predict(gray[y:y+h,x:x+w])  #Our trained model is working here
        # Set the name according to id
        if Id == 1:
            Id = "Manik {0:.2f}%".format(round(100 - confidence, 2))
            flag_id1=1
            # Put text describe who is in the picture
        elif Id == 2 :
            Id = "Chhavi {0:.2f}%".format(round(100 - confidence, 2))
            flag_id2=1
            # Put text describe who is in the picture
        elif Id == 3:
            Id = "Ekanshu {0:.2f}%".format(round(100 - confidence, 2))
            flag_id3=1
        elif Id == 4 :
            Id = "Babanjot {0:.2f}%".format(round(100 - confidence, 2))
            flag_id4=1
            # Put text describe who is in the picture
        else:
            pass
        # Set rectangle around face and name of the person
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(im, str(Id), (x,y-40), font, 1, (0,255,255), 2)
    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 
    # press q to close the program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
if flag_id1==1:
    attendance_id1 += 1
if flag_id2==1:
    attendance_id2 += 1
if flag_id3==1:
    attendance_id3 += 1
if flag_id4==1:
    attendance_id4 += 1
    
print("Attendance - Manik :" ,attendance_id1)
print("Attendance - Chhavi :" ,attendance_id2)
print("Attendance - Ekanshu :" ,attendance_id3)
print("Attendance - Babanjot :" ,attendance_id4)
# Terminate video
cam.release()
# Close all windows
cv2.destroyAllWindows()