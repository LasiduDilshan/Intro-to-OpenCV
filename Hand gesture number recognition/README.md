# Hand Gesture Recognition

This project uses OpenCV and MediaPipe to perform real-time hand gesture recognition. The script captures video from a webcam, detects hand landmarks, and classifies gestures based on the positions and states of the fingers. The recognized gestures include numbers (ONE, TWO, THREE, etc.) and other common hand signals.

## Table of Contents

- [Hand Gesture Recognition](#hand-gesture-recognition)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [How It Works](#how-it-works)
    - [Finger State Detection](#finger-state-detection)
    - [Gesture Recognition](#gesture-recognition)
    - [Machine Learning Methods](#machine-learning-methods)
  - [Code Explanation](#code-explanation)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. **Install Required Libraries:**

   Ensure you have Python installed. Then install the necessary libraries using pip:

   ```bash
   pip install opencv-python mediapipe numpy
   ```

## Usage

Run the script to start the hand gesture recognition:

```bash
python hand_gesture_recognition.py
```

- The script will start capturing video from your default webcam.
- Hand gestures will be recognized and displayed in real-time.
- Press 'q' to exit the video capture and close the window.

## How It Works

### Finger State Detection

The script uses MediaPipe to detect hand landmarks in each video frame. It then calculates the angles between joints of each finger to determine if the finger is open or closed.

- **Thumb:** Uses the angles between the thumb's carpometacarpal (CMC), metacarpophalangeal (MCP), and tip landmarks.
- **Index, Middle, Ring, Pinky Fingers:** Uses the angles between the MCP, proximal interphalangeal (PIP), and tip landmarks.

### Gesture Recognition

Based on the states of the fingers (open or closed), the script recognizes specific hand gestures:
- **FIVE:** All fingers open.
- **FOUR:** Thumb closed, other fingers open.
- **THREE:** Thumb and first two fingers open, others closed.
- **TWO:** Thumb and index finger open, others closed.
- **ONE:** Only the index finger open.
- **YEAH:** Index and middle fingers open.
- **ROCK:** Index and pinky fingers open.
- **SPIDERMAN:** Thumb, index, and pinky fingers open.
- **FIST:** All fingers closed.
- **OK:** Index finger and thumb forming a circle, others open.

### Machine Learning Methods

While this project does not involve training a machine learning model, it leverages pre-trained models from MediaPipe for hand landmark detection. The gesture recognition logic is rule-based, utilizing geometric properties and angles calculated from the detected landmarks. 

- **Hand Landmark Detection:** Utilizes a deep learning model from MediaPipe to detect 21 hand landmarks in real-time.
- **Angle Calculation:** Uses the dot product to calculate angles between finger joints to determine the finger states.

## Code Explanation

The provided Python code performs real-time hand gesture recognition using MediaPipe and OpenCV. The code detects hand landmarks and classifies gestures based on the positions and states of the fingers. Here is a detailed explanation of the code and its components:

#### 1. Import Libraries

The code begins by importing necessary libraries:
```python
import cv2
import mediapipe as mp
import numpy as np
import math
```
- `cv2`: OpenCV library for capturing and processing video.
- `mediapipe`: Library for detecting and tracking hand landmarks.
- `numpy`: Library for numerical operations.
- `math`: Library for mathematical functions.

#### 2. Initialize MediaPipe Hands

Initialize the MediaPipe Hands solution:
```python
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
```
- `static_image_mode=False`: The solution treats the input as a video stream.
- `max_num_hands=1`: The maximum number of hands to detect.
- `min_detection_confidence=0.5`: Minimum confidence value for the detection to be considered successful.
- `min_tracking_confidence=0.5`: Minimum confidence value for the tracking to be considered successful.

#### 3. Helper Functions

Define helper functions for calculating Euclidean distance and checking if the thumb is near the index finger.

##### Euclidean Distance Function

```python
def get_euclidean_distance(a_x, a_y, b_x, b_y):
    return ((a_x - b_x) ** 2 + (a_y - b_y) ** 2) ** 0.5
```
- Calculates the Euclidean distance between two points `(a_x, a_y)` and `(b_x, b_y)`.

##### Thumb Near Index Finger Function

```python
def is_thumb_near_first_finger(point1, point2):
    distance = get_euclidean_distance(point1.x, point1.y, point2.x, point2.y)
    return distance < 0.1
```
- Uses `get_euclidean_distance` to check if the thumb is near the index finger by comparing the distance to a threshold (`0.1`).

##### Angle Calculation Function

```python
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
```
- Calculates the angle between three points (a, b, c) using the dot product and arccosine function.

#### 4. Start Capturing Video

Start capturing video using OpenCV:
```python
cap = cv2.VideoCapture(0)
```
- `0`: Indicates the default camera.

#### 5. Main Loop for Video Processing

The main loop processes each frame of the video to detect hand landmarks and recognize gestures:
```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
```
- Captures each frame from the video stream.
- Converts the frame from BGR to RGB format for MediaPipe processing.
- Processes the RGB frame to detect hand landmarks.

#### 6. Detect Hand Landmarks

If hand landmarks are detected, draw them on the frame and determine the state of each finger:
```python
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
```
- `mp_drawing.draw_landmarks`: Draws the detected hand landmarks and connections on the frame.

#### 7. Determine Finger States

Calculate angles and determine if each finger is open or closed:
```python
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
```
- `calculate_angle`: Calculates the angle between three points to determine if a finger is open or closed.
- Each finger is considered open if the calculated angle is greater than a threshold (160 degrees in this case).

#### 8. Recognize Hand Gestures

Recognize specific hand gestures based on the states of the fingers:
```python
            # Recognize hand gestures.
            if thumb_is_open and first_finger_is_open and second_finger_is_open and third_finger_is_open and fourth_finger_is_open:
                gesture = "FIVE"
            elif not thumb_is_open and first_finger_is_open and second_finger_is_open and third_finger_is_open and fourth_finger_is_open:
                gesture = "FOUR"
            elif thumb_is_open and first_finger_is_open and second_finger_is_open and not third_finger_is_open and not fourth_finger_is_open:
                gesture = "THREE"
            elif thumb_is_open and first_finger_is_open and not second_finger_is_open and not third_finger_is_open and not fourth_finger_is_open:
                gesture = "TWO"
            elif not thumb_is_open and first_finger_is
