import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# NEW → Better dataset path
dataset_path = r"C:\Users\goswa\OneDrive\Documents\emotion detection\train"

emotion_labels = ['angry', 'happy', 'neutral', 'sad', 'surprise']
data = []
labels = []

print("Loading dataset with augmentation...")

for label in emotion_labels:
    folder_path = os.path.join(dataset_path, label)
    for image_file in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, image_file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, (48, 48))
            data.append(resized_image)
            labels.append(emotion_labels.index(label))
        except:
            pass

data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], 48, 48, 1))
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

print("Building Improved CNN Model...")

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(len(emotion_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training New Model...")
model.fit(datagen.flow(X_train, y_train),
          validation_data=(X_test, y_test),
          epochs=25, batch_size=32)

model.save("emotion_model_v2.h5")
print("New Model saved successfully!")

y_pred = np.argmax(model.predict(X_test), axis=1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=emotion_labels))
