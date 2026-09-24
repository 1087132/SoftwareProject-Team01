import cv2

def monitor_classroom():
    # Initialize video capture (0 for the primary camera/webcam)
    cap = cv2.VideoCapture(0)
    
    # Load OpenCV's pre-trained face detection cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Initiating classroom monitoring feed... Press 'q' to terminate.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check camera connection.")
            break
        
        # Convert the video frame to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect student faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Draw bounding boxes and labels around detected students
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Student', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Display the total active headcount on the UI
        headcount_text = f'Active Headcount: {len(faces)}'
        cv2.putText(frame, headcount_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Render the monitoring window
        cv2.imshow('Classroom Vision Monitor', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up hardware resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor_classroom()