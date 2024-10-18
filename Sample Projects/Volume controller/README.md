# Hand Gesture-Controlled Volume Adjustment

This Python script uses computer vision techniques to detect hand gestures via a webcam feed and control system volume accordingly.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. **Install Required Libraries:**

   Ensure you have Python installed. Then install the necessary libraries using pip:

   ```bash
   pip install opencv-python mediapipe pyautogui
   ```

   - `opencv-python`: For capturing and processing webcam frames.
   - `mediapipe`: For hand tracking and landmark detection.
   - `pyautogui`: For simulating keyboard presses to control system volume.

3. **Run the Script:**

   ```bash
   python hand_gesture_volume_control.py
   ```

   This command starts the script. Ensure your webcam is connected and properly configured.

## How It Works

The Python script performs the following steps to control volume based on hand gestures:

1. **Capture Webcam Feed:**
   - Opens a live video stream from the default webcam using OpenCV (`cv2.VideoCapture`).

2. **Process Frames:**
   - Converts each frame from BGR to RGB format to work with Mediapipe.
   - Uses Mediapipe (`mp.solutions.hands`) to detect hand landmarks in the frame.

3. **Detect Hand Gestures:**
   - Analyzes the relative positions of thumb and index finger landmarks to determine hand gestures:
     - "Pointing up": Increases system volume.
     - "Pointing down": Decreases system volume.

4. **Control System Volume:**
   - Uses PyAutoGUI (`pyautogui.press`) to simulate keyboard presses based on detected gestures ('volumeup' and 'volumedown').

5. **Display Feedback:**
   - Draws hand landmarks and connections on the webcam feed using Mediapipe's drawing utilities (`mp.solutions.drawing_utils`).

6. **Exit Condition:**
   - Press 'q' to quit the program and close the webcam feed.

## Notes

- Adjust `min_detection_confidence` and `min_tracking_confidence` parameters in the `Hands` class instantiation (`mp_hands.Hands`) to fine-tune hand detection sensitivity.
- Ensure your Python environment has necessary permissions to interact with system volume controls via PyAutoGUI.

## Dependencies

- Python 3.x
- opencv-python
- mediapipe
- pyautogui
