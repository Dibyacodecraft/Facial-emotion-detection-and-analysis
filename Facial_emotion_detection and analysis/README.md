                                    Real-Time Facial Emotion Detection
                                   -------------------------------------


This project implements two approaches for **real-time facial emotion recognition** using a webcam:

1. CNN-Based Facial Emotion Detection** (in `facial_emotion_detection.py`) using grayscale images and a pre-trained deep learning model.
2. MediaPipe-Based Facial Landmark Emotion Estimation** (in `code.py`) using heuristic rules derived from facial landmark distances.

Both approaches also provide **text-to-speech feedback** of the detected emotion.

---
 Features

✅ Real-time emotion detection from webcam  
✅ Text-to-speech feedback using `pyttsx3`  
✅ CNN model with option for training or using a pre-trained model  
✅ MediaPipe-based rule system without model training  
✅ Emotion logging (for CNN version)  
✅ Emoji and color-coded visualization for different emotions  

---
Requirements
Install all dependencies with:

bash
pip install -r requirements.txt
