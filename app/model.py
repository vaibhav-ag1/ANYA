import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

train_data_dir = 'data/train'
test_data_dir = 'data/test'

emotion_to_int = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

def load_and_preprocess_data(data_dir):
    images = []
    labels = []

    emotions = os.listdir(data_dir)
    for emotion in emotions:
        emotion_dir = os.path.join(data_dir, emotion)
        emotion_label = emotion_to_int[emotion]
        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            img = img / 255.0
            images.append(img)
            labels.append(emotion_label)

    images = np.array(images)
    labels = np.array(labels)

    labels = to_categorical(labels, num_classes=len(emotion_to_int))

    return images, labels

X_train, y_train = load_and_preprocess_data(train_data_dir)
X_test, y_test = load_and_preprocess_data(test_data_dir)

num_classes = 7

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 32

X_train_reshaped = X_train.reshape(-1, 48, 48, 1)
X_test_reshaped = X_test.reshape(-1, 48, 48, 1)

train_generator = datagen.flow(X_train_reshaped, y_train, batch_size=batch_size)

history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) / batch_size, epochs=50,
                    validation_data=(X_test_reshaped, y_test))

loss, accuracy = model.evaluate(X_test_reshaped, y_test)  
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

model.save('fem.h5')
print("Model saved successfully.")