import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to display a "punch" effect
def show_punch_effect(frame, x, y, w, h):
    # Draw a red circle in the center of the face to simulate a punch
    center_x, center_y = x + w // 2, y + h // 2
    cv2.circle(frame, (center_x, center_y), 50, (0, 0, 255), 5)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Loop through each face detected
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        
        # Show punch effect
        show_punch_effect(frame, x, y, w, h)
    
    # Display the result
    cv2.imshow('Face Punching System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
