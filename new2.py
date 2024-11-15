import cv2
import pickle
import numpy as np
import os
"""cv2: This is the OpenCV library used for image and video processing.
pickle: Used to save and load Python objects (in this case, face data and names).
numpy: A library for handling large arrays and matrices of data.
os: A module that helps interact with the operating system, such as checking if files or directories exist"""

# Ensure the 'data' directory exists
if not os.path.exists('data'):
    os.makedirs('data')
    """This block checks if a directory named 'data' exists in the current folder.
If it doesn't exist, it creates that directory to store the face data and name data.
"""

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

"""video = cv2.VideoCapture(0): Opens the default camera (usually the webcam). The number 0 represents the first camera connected to your system.
facedetect: This initializes a face detection model using OpenCV's pre-trained Haar cascade for frontal faces, which is stored in cv2.data.haarcascades.
"""

# Check if the camera opened successfully
if not video.isOpened():
    print("Error: Could not open camera.")
    exit()
"""video.isOpened(): This checks if the camera was opened successfully.
If it wasn’t opened, it prints an error message and stops the program using exit().
"""
faces_data = []  # List to store the face images
i = 0  # Counter for frames
name = input("Enter Your Name: ")  # User input for the name
"""faces_data: This will store the cropped images of faces detected from the video.
i: This is a counter used to keep track of how many frames have been processed.
name: Prompts the user to enter their name, which will be saved along with the face data.
"""
# Start capturing video frames
while True:
    ret, frame = video.read()  # Read a frame from the video
    
    # Check if frame is read correctly
    if not ret:
        print("Error: Could not read frame.")
        break  
    """while True: This starts an infinite loop that will keep capturing video frames until the user stops it.
ret, frame = video.read(): Captures a frame from the video feed. ret is a boolean that tells if the frame was captured successfully, and frame contains the image data of the frame.
if not ret: If a frame couldn’t be read, it prints an error and breaks the loop."""
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame
    """gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY): Converts the color frame (BGR format) to grayscale. Face detection works better in grayscale because it simplifies the image.
    faces = facedetect.detectMultiScale(gray, 1.3, 5): Detects faces in the grayscale image. The parameters 1.3 (scale factor) and 5 (min neighbors) control the accuracy and sensitivity of face detection."""
    
# Process each detected face
    for (x, y, w, h) in faces:
        # Crop and resize the detected face
        crop_img = frame[y:y+h, x:x+w]  
        resized_img = cv2.resize(crop_img, (50, 50))  
        """for (x, y, w, h) in faces: Loops over all the faces detected in the frame. Each face is represented by a rectangle with the top-left corner at (x, y) and width w and height h.
crop_img = frame[y
+h, x
+w]: Crops the face out of the frame using the detected rectangle coordinates.
resized_img = cv2.resize(crop_img, (50, 50)): Resizes the cropped face image to a 50x50 pixel size for consistency."""
        
        # Add face data every 10 frames if less than 100 faces are collected
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        
        i += 1  # Increment the frame counter
        """if len(faces_data) <= 100 and i % 10 == 0: Every 10th frame, if there are fewer than 100 faces collected, the resized face is added to faces_data.
i += 1: Increments the frame counter.
cv2.putText(frame, str(len(faces_data)), ...): Displays the number of collected faces on the video feed.
cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1): Draws a red rectangle around the detected face on the video feed.
"""
        
        # Display face count and draw rectangle around the face
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    
    # Show the frame with face detection
    cv2.imshow("Frame", frame)
    
    # Break the loop if 'q' is pressed or 100 faces are collected
    k = cv2.waitKey(10)
    if k == ord('q') or len(faces_data) == 100:
        break
    """cv2.imshow("Frame", frame): Displays the current frame with the detected faces and rectangle drawn.
cv2.waitKey(10): Waits for a key press for 10 milliseconds.
if k == ord('q') or len(faces_data) == 100: If the 'q' key is pressed or 100 faces have been collected, the loop breaks."""

# Release the video capture object and close windows
video.release()
cv2.destroyAllWindows()
"""video.release(): Releases the video capture object, freeing up the camera.
cv2.destroyAllWindows(): Closes any OpenCV windows that were opened.
"""

# Convert the list of faces into a numpy array
faces_data = np.asarray(faces_data)

# Ensure we have exactly 100 faces before reshaping
if len(faces_data) == 100:
    faces_data = faces_data.reshape(100, -1)
else:
    print(f"Error: Expected 100 faces, but got {len(faces_data)}")
    """faces_data = np.asarray(faces_data): Converts the list of face images into a NumPy array.
if len(faces_data) == 100: Checks if exactly 100 faces were collected.
faces_data.reshape(100, -1): Reshapes the array to have 100 rows (one for each face) and as many columns as needed.
else: If fewer than 100 faces were collected, it prints an error message.
"""

# Handle name storage in 'names.pkl'
names_file = 'data/names.pkl'
if 'names.pkl' not in os.listdir('data'):
    names = [name] * 100  # Create a list with the name repeated 100 times
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    # Load existing names and append the new ones
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
        """if 'names.pkl' not in os.listdir('data'): Checks if a file named names.pkl exists in the data directory.
[name] * 100: Creates a list with the entered name repeated 100 times.
pickle.dump(): Saves the list of names to the file.
pickle.load(): If the file exists, it loads the existing names, appends the new names, and saves the updated list.
"""

# Handle face data storage in 'faces_data.pkl'
faces_data_file = 'data/faces_data.pkl'
if 'faces_data.pkl' not in os.listdir('data'):
    with open(faces_data_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    # Load existing face data and append the new ones
    with open(faces_data_file, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open(faces_data_file, 'wb')as f:
        pickle.dump(faces,f)
        """Similar to name data, this block checks if faces_data.pkl exists. If not, it saves the new face data.
If the file exists, it loads the existing face data, appends the new faces, and saves the combined data back to the file."""