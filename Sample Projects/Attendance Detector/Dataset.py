import cv2
import numpy as np
import os
import pickle

video = cv2.VideoCapture(0)  # 0 for webcam
facedetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # to detect faces
face_data = []
i = 0
name = input("Enter your name: ")

while True:
    ret, frame = video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # color to grayscale
    faces = facedetector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop = frame[y:y+h, x:x+w]
        resized = cv2.resize(crop, (50, 50))
        if len(face_data) < 100 and i % 10 == 0:
            face_data.append(resized)
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        i += 1

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if len(face_data) >= 100:
        break

video.release()
cv2.destroyAllWindows()

# Pickle - save the dataset

face_data = np.asarray(face_data)
face_data = face_data.reshape(face_data.shape[0], -1)

data_path = 'data/'

if not os.path.exists(data_path):
    os.makedirs(data_path)

# Save names
if 'names.pkl' not in os.listdir(data_path):
    names = [name] * 100
    with open(os.path.join(data_path, 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)
else:
    with open(os.path.join(data_path, 'names.pkl'), 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
    with open(os.path.join(data_path, 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)

# Save face data
if 'face_data.pkl' not in os.listdir(data_path):
    with open(os.path.join(data_path, 'face_data.pkl'), 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open(os.path.join(data_path, 'face_data.pkl'), 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)
    with open(os.path.join(data_path, 'face_data.pkl'), 'wb') as f:
        pickle.dump(faces, f)