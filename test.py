
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import win32com.client
from win32com.client import Dispatch
"""KNeighborsClassifier: This is a machine learning model from sklearn used for classification.
cv2: OpenCV library used for handling video and images.
pickle: Helps save and load Python objects (in this case, names and face data).
numpy: Library to work with arrays (face data).
os: Used for file handling (like checking if a file exists).
csv: Used for writing attendance data into CSV files.
time, datetime: For getting the current time and date.
win32com.client: Allows you to access Windows components. Used here for text-to-speech.
Dispatch: Specifically used to control the speech functionality on Windows.
"""

def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

    """This defines a function speak() that takes a string (str1) and makes the computer speak it aloud using the Windows speech API (SAPI.SpVoice).
"""

video=cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
"""video = cv2.VideoCapture(0): Opens the camera (0 means default camera).
facedetect: Loads a pre-trained face detection model (Haar Cascade) from OpenCV for detecting faces."""

with open('data/names.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)
print("FACES shape: ", FACES.shape)
print("LABELS shape: ", len(LABELS))
LABELS = LABELS[:len(FACES)]

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(FACES, LABELS)


"""knn = KNeighborsClassifier(n_neighbors=5): Creates a K-Nearest Neighbors classifier with 5 neighbors.
knn.fit(FACES, LABELS): Trains the KNN model using the loaded face data (FACES) and their corresponding names (LABELS)."""


imgBackground=cv2.imread("data\photo.png")

COL_NAMES = ['NAME', 'TIME']
"""imgBackground: Loads an image called photo.png as a background (this could be a template or a frame overlay).
COL_NAMES: Defines column names for the CSV file where attendance will be saved. The columns will be Name and Time."""
while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    """while True: Starts an infinite loop to continuously capture video frames.
video.read(): Captures a frame from the video.
gray = cv2.cvtColor(...): Converts the color image (frame) to grayscale because face detection works better in grayscale.
facedetect.detectMultiScale(...): Detects faces in the grayscale image. It returns the coordinates of each detected face."""
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        """for (x, y, w, h) in faces: Loops through each detected face in the frame. The face is represented by the rectangle coordinates (x, y, w, h).
crop_img: Crops the detected face from the frame.
resized_img: Resizes the cropped face to a 50x50 pixel image and flattens it (converts the image to a 1D array).
knn.predict(resized_img): Uses the KNN model to predict whose face is in the cropped image.
"""
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        """ts = time.time(): Gets the current timestamp.
date: Converts the timestamp into a readable date (day-month-year format).
timestamp: Converts the timestamp into the current time (hours, minutes, seconds).
"""
        exist=os.path.isfile("data\Attendance_21-10-2024.csv" + date + ".csv")
        
        """os.path.isfile(...): Checks if the attendance file for the current day already exists. If the file exists, it will append to it; if not, it will create a new file.
"""
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        """cv2.rectangle(...): Draws a red rectangle around the detected face.
cv2.putText(...): Displays the predicted name (output from the KNN) above the face.
"""
        attendance=[str(output[0]), str(timestamp)]
        imgBackground= frame
        cv2.imshow("Frame",imgBackground)
        k=cv2.waitKey(1)
        if k==ord('p'):
            speak("Attendance Taken..")
            """attendance: Creates an attendance entry with the predicted name and the current timestamp.
cv2.imshow(...): Displays the current frame with the detected faces and names.
cv2.waitKey(1): Waits for a key press. If the 'p' key is pressed, the speak() function announces "Attendance Taken."""
        #time.sleep(5)
        if exist:
            with open("data\Attendance_21-10-2024.csv" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("data\Attendance_21-10-2024.csv" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

                """if exist: If the attendance file exists, the code appends the new attendance data to it.
else: If the file doesn’t exist, it creates a new file, writes the column names first, and then writes the attendance data."""
            csvfile.close()
            if k==ord('q'):
                break
                video.release();
                cv2.destroyAllWindows()
                """if k == ord('q'): If the 'q' key is pressed, it breaks the loop.
video.release(): Releases the video capture (frees up the camera).
cv2.destroyAllWindows(): Closes all OpenCV windows.
"""