# Hand Tracking with OpenCV and MediaPipe

This repository contains a project for real-time hand tracking using OpenCV and MediaPipe. The project includes three main Python scripts: `Basics.py`, `HandTrackingModule.py`, and `ProjectExample.py`.

## Table of Contents
- [Credits](#credits)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Basics.py](#basicspy)
  - [HandTrackingModule.py](#handtrackingmodulepy)
  - [ProjectExample.py](#projectexamplepy)
- [Explanation of Coordinates](#explanation-of-coordinates)
- [Contributing](#contributing)
- [License](#license)

## Credits:
- All credits for the content in this repository go to [Computer Vision Zone](https://www.computervision.zone/courses/hand-tracking/). The code and concepts are based on the resources provided by Computer Vision Zone for their hand tracking course.
- This repository was created by me solely for educational purposes and to learn about hand tracking for my upcoming projects. Any modifications or additions to the code were made for personal learning and experimentation.

## Overview

This project demonstrates how to perform real-time hand tracking using OpenCV and MediaPipe. It includes functionality for detecting hand landmarks, drawing landmarks, and calculating the positions of specific hand landmarks in each video frame.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/hand-tracking-opencv-mediapipe.git
   cd hand-tracking-opencv-mediapipe
   ```

2. **Install the required libraries:**
   ```bash
   pip install opencv-python mediapipe
   ```

## Usage

### Basics.py

**Purpose**: Demonstrates basic hand detection and landmark drawing using the MediaPipe Hands module.

**Code Explanation**:
- Initializes video capture from the webcam.
- Uses MediaPipe to detect hand landmarks and draw them on each frame.
- Prints the ID and coordinates of each landmark.

**How to Run**:
```bash
python Basics.py
```

### HandTrackingModule.py

**Purpose**: Defines a reusable hand detection module encapsulated in the `handDetector` class. Provides methods to detect hands and find the positions of hand landmarks.

**Code Explanation**:
- **`__init__`**: Initializes the MediaPipe Hands model.
- **`findHands`**: Detects hands and optionally draws landmarks on the image.
- **`findPosition`**: Extracts the positions of landmarks and optionally draws circles on them.

**How to Run**:
The script includes a main function for testing purposes. To run the test:
```bash
python HandTrackingModule.py
```

### ProjectExample.py

**Purpose**: Uses the `handDetector` class from `HandTrackingModule.py` to perform real-time hand detection and tracking.

**Code Explanation**:
- Captures video from the webcam.
- Uses the `handDetector` to detect hands and draw landmarks.
- Extracts and prints the position of the thumb tip (landmark ID `4`).

**How to Run**:
```bash
python ProjectExample.py
```

## Explanation of Coordinates

The coordinates printed and displayed in the scripts correspond to specific hand landmarks detected by MediaPipe. The landmarks have the following IDs:

```
0: Wrist
1: Thumb CMC
2: Thumb MCP
3: Thumb IP
4: Thumb tip
5: Index finger MCP
6: Index finger PIP
7: Index finger DIP
8: Index finger tip
9: Middle finger MCP
10: Middle finger PIP
11: Middle finger DIP
12: Middle finger tip
13: Ring finger MCP
14: Ring finger PIP
15: Ring finger DIP
16: Ring finger tip
17: Pinky finger MCP
18: Pinky finger PIP
19: Pinky finger DIP
20: Pinky finger tip
```

Each coordinate is in the form `[id, x, y]`, where:
- `id`: The landmark ID.
- `x`: The x-coordinate of the landmark in the image.
- `y`: The y-coordinate of the landmark in the image.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
```

### Detailed Explanation of Each Script

#### Basics.py

- **Video Capture Initialization**: 
  ```python
  cap = cv2.VideoCapture(1)
  ```
  Initializes video capture from the webcam at index 1.

- **MediaPipe Hands Initialization**:
  ```python
  mpHands = mp.solutions.hands
  hands = mpHands.Hands()
  mpDraw = mp.solutions.drawing_utils
  ```
  Initializes the MediaPipe Hands module and drawing utilities.

- **Main Loop**:
  - Captures each frame from the webcam.
  - Converts the frame to RGB format.
  - Processes the frame to detect hand landmarks.
  - Draws circles on detected landmarks and prints their coordinates.
  - Calculates and displays the frames per second (FPS).

#### HandTrackingModule.py

- **Initialization (`__init__` method)**:
  ```python
  class handDetector:
      def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
          self.mode = mode
          self.maxHands = maxHands
          self.detectionCon = detectionCon
          self.trackCon = trackCon

          self.mpHands = mp.solutions.hands
          self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                          max_num_hands=self.maxHands,
                                          min_detection_confidence=self.detectionCon,
                                          min_tracking_confidence=self.trackCon)
          self.mpDraw = mp.solutions.drawing_utils
  ```

- **Method: `findHands`**:
  ```python
  def findHands(self, img, draw=True):
      imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      self.results = self.hands.process(imgRGB)
      if self.results.multi_hand_landmarks:
          for handLms in self.results.multi_hand_landmarks:
              if draw:
                  self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
      return img
  ```

- **Method: `findPosition`**:
  ```python
  def findPosition(self, img, handNo=0, draw=True):
      lmList = []
      if self.results.multi_hand_landmarks:
          myHand = self.results.multi_hand_landmarks[handNo]
          for id, lm in enumerate(myHand.landmark):
              h, w, c = img.shape
              cx, cy = int(lm.x * w), int(lm.y * h)
              lmList.append([id, cx, cy])
              if draw:
                  cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
      return lmList
  ```

#### ProjectExample.py

- **Video Capture Initialization**:
  ```python
  cap = cv2.VideoCapture(0)
  ```

- **Hand Detector Initialization**:
  ```python
  detector = htm.handDetector()
  ```

- **Main Loop**:
  - Captures each frame from the webcam.
  - Uses the `handDetector` to find hands and draw landmarks.
  - Extracts the positions of landmarks.
  - Prints the position of the thumb tip (landmark ID `4`).
  - Calculates and displays the frames per second (FPS).

### How to Use the Codes

1. **Basics.py**:
   - Run this script to see basic hand detection and landmark drawing. It will display the webcam feed with detected hand landmarks and print the coordinates of each landmark.

2. **HandTrackingModule.py**:
   - This script defines the `handDetector` class. You can use this class in other scripts to perform hand detection and extract landmark positions.

3. **ProjectExample.py**:
   - This script uses the `handDetector` class to demonstrate real-time hand tracking. It captures the webcam feed, detects hands, draws landmarks, and prints the coordinates of the thumb tip.

By following these instructions, you can set up and run the hand tracking project, understand the coordinates and landmarks, and use the provided modules to extend the functionality as needed.
