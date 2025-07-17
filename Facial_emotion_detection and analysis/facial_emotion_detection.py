import os
import cv2
import numpy as np
import pandas as pd
import pyttsx3
import time
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Set working directory to where 'archive' folder is present
os.chdir("C:/Users/Deeksha/Desktop/FacialEmotionProject")  # Update this path to your actual project directory
print("Current Directory:", os.getcwd())

# Load images and labels
def load_data(data_dir):
    images = []
    labels = []
    for expression in os.listdir(data_dir):
        expression_path = os.path.join(data_dir, expression)
        if os.path.isdir(expression_path):
            for img_file in os.listdir(expression_path):
                img_path = os.path.join(expression_path, img_file)
                try:
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (48, 48))
                    images.append(image)
                    labels.append(expression)
                except Exception as e:
                    print(f"Error loading: {img_path} - {e}")
    return np.array(images), np.array(labels)

# Load training and testing data
try:
    X_train, y_train = load_data("archive/train")
    X_test, y_test = load_data("archive/test")
except FileNotFoundError:
    print("Error: 'archive/train' or 'archive/test' directory not found. Ensure the dataset is in the correct path.")
    exit()

# Reshape and normalize images
X_train = X_train.reshape(-1, 48, 48, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 48, 48, 1).astype('float32') / 255.0

# Encode labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# One-hot encode
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Emotion Classes Detected:", list(encoder.classes_))

# Define CNN model (example architecture; replace with your trained model if available)
def create_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load or create model
model_path = "emotion_model_best.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")
else:
    print(f"Model file {model_path} not found. Creating new model (requires training).")
    model = create_model()
    # Note: You need to train the model here or provide a pre-trained model file
    # Example: model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))
    print("Warning: New model created but not trained. Please train the model or provide a pre-trained model file.")

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.6)

# Function to speak emotion with time interval control
def speak_emotion(emotion, last_spoken_time, interval=5):
    current_time = time.time()
    if current_time - last_spoken_time >= interval:
        engine.say(f"You look {emotion.lower()}")
        engine.runAndWait()
        return current_time
    return last_spoken_time

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Error: Failed to load Haar Cascade classifier.")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("🎥 Real-time Emotion Detection Started... Press 'q' to quit.")

# Emotion logging
emotion_log = []
last_spoken_time = time.time()
spoken_interval = 5
frame_count = 0
frame_gap = 7
last_emotion = None

# Webcam loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_normalized = roi_resized / 255.0
        roi_input = np.reshape(roi_normalized, (1, 48, 48, 1))

        if frame_count % frame_gap == 0:
            prediction = model.predict(roi_input, verbose=0)
            emotion_index = np.argmax(prediction)
            predicted_emotion = encoder.classes_[emotion_index]
            confidence = float(np.max(prediction)) * 100

            # Log emotion
            emotion_log.append([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), predicted_emotion, confidence])

            # Speak only if emotion changed or enough time has passed
            if predicted_emotion != last_emotion:
                last_spoken_time = speak_emotion(predicted_emotion, last_spoken_time, spoken_interval)
                last_emotion = predicted_emotion

            # Debugging
            print(f"Raw Prediction Probabilities: {prediction}")

        # Draw rectangle and label
        color = (0, 255, 0) if predicted_emotion == 'Happy' else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{predicted_emotion} ({confidence:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Real-Time Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save emotion log to CSV
if emotion_log:
    log_df = pd.DataFrame(emotion_log, columns=["Timestamp", "Emotion", "Confidence"])
    log_filename = f"emotion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_df.to_csv(log_filename, index=False)
    print(f"\n📁 Emotion log saved as: {log_filename}")

# Release resources
cap.release()
cv2.destroyAllWindows()
engine.stop()