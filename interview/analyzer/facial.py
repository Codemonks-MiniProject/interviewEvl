import cv2
import numpy as np
from deepface import DeepFace

# Emotion to confidence weights
EMOTION_CONFIDENCE = {
    'happy': 0.9,
    'neutral': 0.7,
    'surprise': 0.6,
    'sad': 0.3,
    'angry': 0.2,
    'fear': 0.2,
    'disgust': 0.1
}

def analyze_facial_emotions(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                confidence = EMOTION_CONFIDENCE.get(emotion, 0.5)  # default if unknown
                scores.append(confidence * 100)
            except Exception as e:
                continue  # Skip failed frames

        frame_count += 1

    cap.release()
    return round(float(np.mean(scores)), 2) if scores else 0.0