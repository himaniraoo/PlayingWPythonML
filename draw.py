import cv2
import numpy as np
import mediapipe as mp
import math

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Create a blank white image as the virtual "paper"
canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255

# Variables to store previous fingertip positions for drawing
prev_x, prev_y = None, None

# Helper function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Function to check if the hand is making a fist
def is_fist(landmarks, w, h):
    # Landmarks for fingertips and MCP joints (knuckles)
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_mcps = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]
    
    # Check if each fingertip is close to its corresponding MCP (finger is curled)
    for tip, mcp in zip(finger_tips, finger_mcps):
        tip_x = int(landmarks.landmark[tip].x * w)
        tip_y = int(landmarks.landmark[tip].y * h)
        mcp_x = int(landmarks.landmark[mcp].x * w)
        mcp_y = int(landmarks.landmark[mcp].y * h)
        distance = calculate_distance((tip_x, tip_y), (mcp_x, mcp_y))

        if distance > 40:  # If the distance is large, the finger is extended
            return False

    return True

# Start capturing video
cap = cv2.VideoCapture(0)

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
            # Get landmarks for index finger tip (8) and MCP (5)
            x_tip = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            y_tip = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            
            x_mcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * w)
            y_mcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * h)
            
            # Detect if the hand is making a fist (used to clear the canvas)
            if is_fist(hand_landmarks, w, h):
                # Clear the canvas
                canvas[:] = 255
                prev_x, prev_y = None, None  # Reset previous points to avoid unintended lines
            else:
                # Only draw if the index finger is extended
                distance = calculate_distance((x_tip, y_tip), (x_mcp, y_mcp))
                if distance > 40:  # Threshold for determining if the index finger is extended
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x_tip, y_tip), (0, 0, 0), 5)  # Draw in black

                    prev_x, prev_y = x_tip, y_tip
                else:
                    prev_x, prev_y = None, None  # Reset if finger is not extended

            # Draw hand landmarks on the frame (optional, for visualization)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the live video and the virtual canvas
    cv2.imshow("Air Writing - Camera", frame)
    cv2.imshow("Air Writing - Canvas", canvas)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
