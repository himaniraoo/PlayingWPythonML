import cv2
import numpy as np
import mediapipe as mp
import math

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Start capturing video
cap = cv2.VideoCapture(0)

# Initial zoom level
zoom_factor = 1.0
max_zoom = 10.0  # Maximum zoom level
min_zoom = 0.1  # Minimum zoom level

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    h, w, c = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get landmarks for thumb tip (4) and index finger tip (8)
            thumb_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w)
            thumb_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h)
            index_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

            # Calculate the distance between the thumb tip and index finger tip
            distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

            # Map the distance to zoom factor
            # You can adjust these constants to change sensitivity
            zoom_factor = max(min_zoom, min(max_zoom, 1.0 + (distance - 50) / 100))

            # Resize the frame based on the zoom factor
            zoomed_frame = cv2.resize(frame, (int(w * zoom_factor), int(h * zoom_factor)))
            # If the zoomed frame is larger than the original, crop it
            if zoomed_frame.shape[0] > h or zoomed_frame.shape[1] > w:
                start_x = max(0, int((zoomed_frame.shape[1] - w) / 2))
                start_y = max(0, int((zoomed_frame.shape[0] - h) / 2))
                zoomed_frame = zoomed_frame[start_y:start_y + h, start_x:start_x + w]

            # Display the zoomed frame
            cv2.imshow("Zoom Control - Camera", zoomed_frame)

            # Draw hand landmarks on the frame (optional, for visualization)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
