
import cv2
import numpy as np

# Using haarcascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 0: Webcam | Change 0 to appropriate webcam index
cap = cv2.VideoCapture(0)

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detecting faces in the frame
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	# Drawing rectangle on frame around faces
	for x, y, w, h in faces:
		cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

	cv2.imshow('Image', img)
	k = cv2.waitKey(30)

	# Break of Esc is pressed
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
