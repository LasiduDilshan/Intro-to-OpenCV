import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points.
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Function to calculate Euclidean distance between two points.
def get_euclidean_distance(a_x, a_y, b_x, b_y):
    return ((a_x - b_x) ** 2 + (a_y - b_y) ** 2) ** 0.5

# Function to check if thumb is near the index finger.
def is_thumb_near_first_finger(point1, point2):
    distance = get_euclidean_distance(point1.x, point1.y, point2.x, point2.y)
    return distance < 0.1

# Start capturing video.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # Get coordinates.
            thumb_cmc = np.array([landmarks[mp_hands.HandLandmark.THUMB_CMC].x, landmarks[mp_hands.HandLandmark.THUMB_CMC].y])
            thumb_mcp = np.array([landmarks[mp_hands.HandLandmark.THUMB_MCP].x, landmarks[mp_hands.HandLandmark.THUMB_MCP].y])
            thumb_ip = np.array([landmarks[mp_hands.HandLandmark.THUMB_IP].x, landmarks[mp_hands.HandLandmark.THUMB_IP].y])
            thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x, landmarks[mp_hands.HandLandmark.THUMB_TIP].y])

            index_finger_mcp = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y])
            index_finger_pip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y])
            index_finger_dip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y])
            index_finger_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])

            middle_finger_pip = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y])
            middle_finger_dip = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y])
            middle_finger_tip = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])

            ring_finger_pip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y])
            ring_finger_dip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y])
            ring_finger_tip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y])

            pinky_finger_pip = np.array([landmarks[mp_hands.HandLandmark.PINKY_PIP].x, landmarks[mp_hands.HandLandmark.PINKY_PIP].y])
            pinky_finger_dip = np.array([landmarks[mp_hands.HandLandmark.PINKY_DIP].x, landmarks[mp_hands.HandLandmark.PINKY_DIP].y])
            pinky_finger_tip = np.array([landmarks[mp_hands.HandLandmark.PINKY_TIP].x, landmarks[mp_hands.HandLandmark.PINKY_TIP].y])

            # Calculate angles to determine if fingers are open or closed.
            thumb_angle = calculate_angle(thumb_cmc, thumb_mcp, thumb_tip)
            index_angle = calculate_angle(index_finger_mcp, index_finger_pip, index_finger_tip)
            middle_angle = calculate_angle(index_finger_mcp, middle_finger_pip, middle_finger_tip)
            ring_angle = calculate_angle(index_finger_mcp, ring_finger_pip, ring_finger_tip)
            pinky_angle = calculate_angle(index_finger_mcp, pinky_finger_pip, pinky_finger_tip)

            thumb_is_open = thumb_angle > 160
            first_finger_is_open = index_angle > 160
            second_finger_is_open = middle_angle > 160
            third_finger_is_open = ring_angle > 160
            fourth_finger_is_open = pinky_angle > 160

            # Recognize hand gestures.
            if thumb_is_open and first_finger_is_open and second_finger_is_open and third_finger_is_open and fourth_finger_is_open:
                gesture = "FIVE"
            elif not thumb_is_open and first_finger_is_open and second_finger_is_open and third_finger_is_open and fourth_finger_is_open:
                gesture = "FOUR"
            elif thumb_is_open and first_finger_is_open and second_finger_is_open and not third_finger_is_open and not fourth_finger_is_open:
                gesture = "THREE"
            elif thumb_is_open and first_finger_is_open and not second_finger_is_open and not third_finger_is_open and not fourth_finger_is_open:
                gesture = "TWO"
            elif not thumb_is_open and first_finger_is_open and not second_finger_is_open and not third_finger_is_open and not fourth_finger_is_open:
                gesture = "ONE"
            elif not thumb_is_open and first_finger_is_open and second_finger_is_open and not third_finger_is_open and not fourth_finger_is_open:
                gesture = "YEAH"
            elif not thumb_is_open and first_finger_is_open and not second_finger_is_open and not third_finger_is_open and fourth_finger_is_open:
                gesture = "ROCK"
            elif thumb_is_open and first_finger_is_open and not second_finger_is_open and not third_finger_is_open and fourth_finger_is_open:
                gesture = "SPIDERMAN"
            elif not thumb_is_open and not first_finger_is_open and not second_finger_is_open and not third_finger_is_open and not fourth_finger_is_open:
                gesture = "FIST"
            elif (not first_finger_is_open and second_finger_is_open and third_finger_is_open and fourth_finger_is_open and 
                  is_thumb_near_first_finger(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP], hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])):
                gesture = "OK"
            else:
                gesture = "UNKNOWN"

            # Display the gesture on the frame.
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame.
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on pressing 'q'.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows.
cap.release()
cv2.destroyAllWindows()
