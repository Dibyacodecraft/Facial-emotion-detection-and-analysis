import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
from queue import Queue

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Queue and threading for speech
speech_queue = Queue()
stop_signal = threading.Event()

def speech_loop():
    while not stop_signal.is_set():
        if not speech_queue.empty():
            msg = speech_queue.get()
            engine.say(msg)
            engine.runAndWait()
        time.sleep(0.1)

def speak_emotion(msg):
    if speech_queue.empty():
        speech_queue.put(msg)

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
drawing_utils = mp.solutions.drawing_utils

# Facial landmark sets
LIPS = mp_face_mesh.FACEMESH_LIPS
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

emotions = {
    "happy": {"emoji": "😊", "color": (0, 255, 0), "msg": "You look happy!"},
    "sad": {"emoji": "😢", "color": (255, 0, 0), "msg": "You seem sad."},
    "angry": {"emoji": "😠", "color": (0, 0, 255), "msg": "You look angry!"},
    "surprise": {"emoji": "😲", "color": (0, 255, 255), "msg": "You're surprised!"},
    "neutral": {"emoji": "😐", "color": (200, 200, 200), "msg": "You look neutral."},
    "fear": {"emoji": "😨", "color": (255, 140, 0), "msg": "You look scared."},
    "disgust": {"emoji": "🤢", "color": (138, 43, 226), "msg": "You're feeling disgusted."}
}

def distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def get_emotion(landmarks):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]
    iris_left = landmarks[468]

    face_width = distance(landmarks[234], landmarks[454])
    mouth_open = distance(top_lip, bottom_lip) / face_width
    mouth_stretch = distance(left_mouth, right_mouth) / face_width
    eye_open = (distance(left_eye_top, left_eye_bottom) + distance(right_eye_top, right_eye_bottom)) / (2 * face_width)
    eye_center_y = (left_eye_top.y + left_eye_bottom.y + right_eye_top.y + right_eye_bottom.y) / 4
    sad_offset = iris_left.y - eye_center_y

    if mouth_stretch > 0.40 and mouth_open < 0.06:
        return "happy"
    elif mouth_open >= 0.12:
        return "surprise"
    elif 0.06 < mouth_open < 0.12:
        return "fear"
    elif sad_offset > 0.01 and eye_open < 0.04:
        return "sad"
    elif mouth_open < 0.03 and eye_open < 0.08 and mouth_stretch < 0.38:
        return "disgust"
    elif eye_open > 0.096 and mouth_open < 0.06:
        return "angry"
    else:
        return "neutral"

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_emotion = ""
frame_count = 0

# Start speaker thread
speaker_thread = threading.Thread(target=speech_loop, daemon=True)
speaker_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    avatar_canvas = np.zeros_like(frame)

    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmark_list = face_landmarks.landmark

        if frame_count % 2 == 0:  # Faster detection
            emotion = get_emotion(landmark_list)

            if emotion != prev_emotion:
                prev_emotion = emotion
                speak_emotion(emotions[emotion]["msg"])
        else:
            emotion = prev_emotion

        color = emotions[emotion]["color"]
        emoji = emotions[emotion]["emoji"]

        # Draw face mesh on avatar canvas
        drawing_utils.draw_landmarks(
            avatar_canvas, face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION, None,
            drawing_utils.DrawingSpec(color=color, thickness=1, circle_radius=1)
        )

        # Draw iris points
        for idx in LEFT_IRIS + RIGHT_IRIS:
            pt = landmark_list[idx]
            cx, cy = int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])
            cv2.circle(avatar_canvas, (cx, cy), 2, (0, 255, 255), -1)

        # Show emotion text
        cx, cy = int(landmark_list[1].x * frame.shape[1]), int(landmark_list[1].y * frame.shape[0])
        cv2.putText(avatar_canvas, f"{emotion.upper()} {emoji}", (cx - 40, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    frame_count += 1

    # Combine original + avatar canvas
    combined = np.hstack((frame, avatar_canvas))
    cv2.imshow("Webcam | Avatar Emotion", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# Cleanup
stop_signal.set()
speaker_thread.join()
cap.release()
cv2.destroyAllWindows()