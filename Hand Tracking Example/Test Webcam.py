import cv2

cap = cv2.VideoCapture(0)  # Change to 1 if you have multiple cameras

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    cv2.imshow("Webcam Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
