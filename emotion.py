import cv2
from fer import FER

# Create an emotion detector object
emotion_detector = FER(mtcnn=True)

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read frame-by-frame from the webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Use the FER library to detect emotions in the frame
    result = emotion_detector.detect_emotions(frame)

    # If emotions are detected, extract and display them
    if result:
        for face in result:
            emotions = face["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            confidence = emotions[top_emotion]

            # Draw a rectangle around the detected face
            (x, y, w, h) = face["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{top_emotion}: {confidence:.2f}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame with detected emotions
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
