import cv2 # We will import openCV library for image processing, opening the webcam etc
import os #Os is required for managing files like directories
import numpy as np #Numpy is basically used for matrix operations
from PIL import Image #PIL is Python Image Library
#Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
# Using Local Binary Patterns Histograms for face recognization since it's quite accurate than the rest
model = cv2.face.LBPHFaceRecognizer_create()
# For detecting the faces in each frame we will use Haarcascade Frontal Face default classifier of OpenCV
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
#method getting the images and label data
def getImagesAndLabels(path):
    # Getting all file paths
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    #empty face sample initialised
    faceSamples=[]    
    # IDS for each individual
    ids = []
    # Looping through all the file path
    for imagePath in imagePaths:
        # converting image to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        # converting PIL image to numpy array using array() method of numpy
        #L is for converting it to grayscale/black and white
        img_numpy = np.array(PIL_img,'uint8')
        # Getting the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        #Splitting at .
        #-1 means from back
        # Getting the face from the training images
        faces = detector.detectMultiScale(img_numpy)
        # Looping for each face and appending it to their respective IDs
        for (x,y,w,h) in faces:
            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            # Add the ID to IDs
            ids.append(id)
            #LAbels
    # Passing the face array and IDs array 
    return faceSamples,ids
# Getting the faces and IDs
faces,ids = getImagesAndLabels('training_data')
# Training the model using the faces and IDs
model.train(faces, np.array(ids))
# Saving the model into s_model.yml
assure_path_exists('saved_model/')
model.write('saved_model/s_model.yml')
print("-------------Model Trained Successfully-------------")