# from datetime import datetime
import datetime
import os
import cv2
import re
import urllib.request
import firebase_admin
# from firebase import Firebase
from firebase_admin import credentials, storage
import numpy as np
from PIL import Image

DOWNLOAD_FOLDER = "downloaded_images"
DATASET_NEW_FOLDER = "dataset_new"

# Fetch the service account key JSON file contents
cred = credentials.Certificate("hackathonproject-b82e8-firebase-adminsdk-hfy9s-e4e53ddd7d.json")

# Initialize the app with a service account, granting admin privileges
app = firebase_admin.initialize_app(cred, {
    'storageBucket': 'hackathonproject-b82e8.appspot.com',
}, name='storage')



# 1. Download the image files from cloud
def downloadDataForDataSet():
    bucket = storage.bucket(app=app)
    blob_list = list(bucket.list_blobs())
    for blob in blob_list:
        fileName = blob.name
        if "ForRecognition" in fileName:
            print(fileName)
            fileUrl = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
            print(fileUrl)
            urllib.request.urlretrieve(fileUrl, DOWNLOAD_FOLDER+"/"+fileName)


#downloadDataForDataSet()

import csv

csv_reader = {}
with open('FaceDetails.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    print("Dict created.")


def getFaceidFromDictionary(name):
    # print ("CSV nameid value for : "+ name + ": " + csv_reader[name])
    try:
        line_count = 0
        with open('FaceDetails.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            # print("getFaceidFromDictionary: Dict created.")
            line_count = 1
            for row in csv_reader:
                # print("getFaceidFromDictionary: checking name: " + row["name"])
                if (row["name"] == name):
                    return row["face_id"]
                line_count += 1
        # if entry not found, add a new entry into csv file
        with open('FaceDetails.csv', mode='a+', newline='') as csv_file:
            csv_columns = ["name", "face_id", "email"]
            # csv_writer.writeheader()
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            csv_writer.writerow({'name': name, 'face_id': line_count, 'email': 'hello@hcl.com'})

            print(f'Processed {line_count} lines.')
    except IOError:
        print("IOERROR in opening CSV")

    return 1


def getFaceid(filename_key):
    filename, extention = os.path.splitext(filename_key)  # removing extension from filename
    # print(filename)
    match = re.match(r"([a-z]+)([0-9]+)", filename, re.I)
    name_key = ""
    if match:
        name_key = match.group(1)
    else:
        print("ERROR - input name should have number: e.g. vijesh1.jpg")
    print("getFaceid: name:" + name_key)
    face_id = getFaceidFromDictionary(name_key)
    return face_id

# ******************************************************************************************
def CREATE_DATASET():
    # 2. Create data set for training
    # Downloaded data is parsed and data set created.
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    fileList = os.listdir(DOWNLOAD_FOLDER)
    count = 1
    # fileList = []
    for filename in fileList:
        print("FILENAME:: " + filename)
        img = cv2.imread(DOWNLOAD_FOLDER+"/"+filename)
        img = cv2.flip(img, -1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        face_id = getFaceid(filename)

        for (x, y, w, h) in faces:
            print("Face detected in file: " + filename)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Save the captured image into the datasets folder
            print ("creating dataset file: "+ DATASET_NEW_FOLDER + "/User." + str(face_id) + '.' + str(count) + ".jpg")
            cv2.imwrite(DATASET_NEW_FOLDER + "/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
            count += 1
            # cv2.imshow('image', img)

CREATE_DATASET()

# *******************************************************************************************************
# Creating model

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


# function to get the images and label data
def getImagesAndLabels(path):
    print("inside getImageAndLabels")
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return faceSamples, ids

def CREATE_MODEL():
    # Path for face image database
    path = DATASET_NEW_FOLDER
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

CREATE_MODEL()

# ***********************************************************************************************************
# Trying to recognize faces
RECOG_FOLDER = "ForRecognition"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
# iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'RASHMIKA', 'KATRINA', 'SHRUTHI', 'Ilza', 'Z', 'W']
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)


def getNameFromId(id):
    return names[id]


def RecogFace(filePath):
    # print("Inside Recogfile: " + filePath)
    img = cv2.imread(filePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            # id = names[id]
            faceName = getNameFromId(id)
            confidence = "  {0}%".format(round(100 - confidence))
            print(filePath + ":  " + faceName)
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            print(filePath + ":  Unknown")


# 1. Download the image files from cloud
def downloadRecogFilesFromFirebase():
    try:
        bucket = storage.bucket(app=app)
        blob_list = list(bucket.list_blobs())
        for blob in blob_list:
            fileName = blob.name
            if RECOG_FOLDER in fileName:
                print(fileName)
                fileUrl = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
                print(fileUrl)
                urllib.request.urlretrieve(fileUrl, fileName)
    except:
        print("ERROR in downloading file")
    finally:
        print("FINALLY")


#downloadRecogFilesFromFirebase()

fileList = os.listdir(RECOG_FOLDER)
for filename in fileList:
    RecogFace(RECOG_FOLDER + "/" + filename)
