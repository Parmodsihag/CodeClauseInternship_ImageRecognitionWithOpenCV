import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam for video capture
video_capture = cv2.VideoCapture(0)

# Loop through video frames
while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Exit the loop if the video capture fails
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 
                                         scaleFactor=1.1,  # Adjust for different face sizes
                                         minNeighbors=5,  # Minimum number of neighboring detections
                                         minSize=(30, 30)) # Minimum face size to detect

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
