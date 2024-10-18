import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

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

            # Determine the state of each finger.
            thumb_is_open = False
            first_finger_is_open = False
            second_finger_is_open = False
            third_finger_is_open = False
            fourth_finger_is_open = False

            pseudo_fix_key_point = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
            if (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x < pseudo_fix_key_point and 
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < pseudo_fix_key_point):
                thumb_is_open = True

            pseudo_fix_key_point = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y < pseudo_fix_key_point and 
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < pseudo_fix_key_point):
                first_finger_is_open = True

            pseudo_fix_key_point = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            if (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y < pseudo_fix_key_point and 
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < pseudo_fix_key_point):
                second_finger_is_open = True

            pseudo_fix_key_point = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
            if (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y < pseudo_fix_key_point and 
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < pseudo_fix_key_point):
                third_finger_is_open = True

            pseudo_fix_key_point = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
            if (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y < pseudo_fix_key_point and 
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < pseudo_fix_key_point):
                fourth_finger_is_open = True

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
