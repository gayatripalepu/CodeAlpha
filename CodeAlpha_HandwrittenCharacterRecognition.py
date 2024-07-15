import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np

# Step 1: Load and Preprocess the Data
(train_imgs, train_lbls), (test_imgs, test_lbls) = mnist.load_data()
train_imgs = train_imgs.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_imgs = test_imgs.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_lbls = to_categorical(train_lbls)
test_lbls = to_categorical(test_lbls)

# Step 2: Build the Model
digit_recognition_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
digit_recognition_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
digit_recognition_model.fit(train_imgs, train_lbls, epochs=5, batch_size=64, validation_split=0.2)

# Step 4: Evaluate the Model
loss, accuracy = digit_recognition_model.evaluate(test_imgs, test_lbls)
print(f'Test accuracy: {accuracy}')

# Step 5: Segment Characters in an Image
def extract_characters(image_of_text):
    # Convert to grayscale and apply thresholding
    grayscale_img = cv2.cvtColor(image_of_text, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding boxes
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])  # Sort left-to-right

    # Extract and preprocess individual characters
    char_imgs = []
    for box in bounding_boxes:
        x, y, w, h = box
        char_img = grayscale_img[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (28, 28))
        char_img = char_img.reshape((28, 28, 1)).astype('float32') / 255
        char_imgs.append(char_img)
    
    return np.array(char_imgs)
