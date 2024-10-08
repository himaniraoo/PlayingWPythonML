import cv2
import os
import mediapipe as mp

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Brightness adjustment function using AppleScript
def adjust_brightness(direction):
    try:
        if direction == "up":
            # Increase brightness using AppleScript
            os.system("osascript -e 'tell application \"System Events\" to key code 144'")
        elif direction == "down":
            # Decrease brightness using AppleScript
            os.system("osascript -e 'tell application \"System Events\" to key code 145'")
        print(f"Adjusted brightness {direction}")

    except Exception as e:
        print(f"Error adjusting brightness: {e}")

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Hand gesture recognition
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame from camera.")
            break

        # Convert the BGR image to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip the image for a mirror view
        image = cv2.flip(image, 1)

        # Process the frame for hand detection
        result = hands.process(image)

        # Convert back to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect hand gestures using landmarks (index finger tip vs wrist)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # If the index finger is above the wrist, increase brightness
                if index_finger_tip.y < wrist.y:
                    adjust_brightness("up")
                # If the index finger is below the wrist, decrease brightness
                else:
                    adjust_brightness("down")

        # Display the frame
        cv2.imshow("Brightness Control", image)

        # Wait for 'q' to be pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
