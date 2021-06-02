
import cv2
import numpy as np
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.vgg19 import VGG19
# from keras import Sequential
from keras.models import load_model
# from keras.layers import Dense, Flatten
# import warnings
# warnings.filterwarnings("ignore")

dataset_path = 'Face Mask Dataset/'
train_path = dataset_path + 'Train'
test_path = dataset_path + 'Test'
validation_path = dataset_path + 'Validation'

img_size = (128, 128)
batch_size = 32

# To train the model again, uncomment and run the lines below
'''
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(train_path, target_size=img_size, batch_size=batch_size, class_mode='binary', color_mode='rgb')
test_generator = datagen.flow_from_directory(test_path, target_size=img_size, batch_size=batch_size, class_mode='binary', color_mode='rgb')
validation_generator = datagen.flow_from_directory(validation_path, target_size=img_size, batch_size=batch_size, class_mode='binary', color_mode='rgb')

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
for layer in vgg19.layers:
	layer.trainable = False

model = Sequential([
	vgg19,
	Flatten(),
	Dense(1, activation='sigmoid')
])
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
history = model.fit_generator(generator=train_generator, steps_per_epoch=len(train_generator)//batch_size, epochs=20, validation_data=validation_generator, validation_steps=len(validation_generator)//batch_size)
model.save('maskModel.h5')
'''

# Loading the trained model from memory
model = load_model('maskModel.h5')

# # Using haarcascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 0: Webcam | Change 0 to appropriate webcam index
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detecting faces in the frame
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	mask_label = {0: 'MASK', 1: 'NO MASK'}

	for x, y, w, h in faces:

		# Drawing rectangle on frame around faces


		#
		cropped_img = img[y:y+h, x:x+w]
		resized_img = cv2.resize(cropped_img, img_size)
		normalized_img = resized_img/255.0
		reshaped_img = np.reshape(normalized_img, (1, img_size[0], img_size[1], 3))
		reshaped_img = np.vstack([reshaped_img])
		mask_result = round(model.predict(reshaped_img)[0][0])

		if mask_result:
			display_color = (0, 0, 255)
		else:
			display_color = (0, 255, 0)

		cv2.rectangle(img, (x, y), (x + w, y + h), display_color, 2)
		cv2.putText(img, mask_label[mask_result], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)

	cv2.imshow('Image', img)
	k = cv2.waitKey(30)

	# Break of Esc is pressed
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
