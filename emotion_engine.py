from pathlib import Path
import json

import cv2
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import img_to_array


BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
MODEL_PATH = BASE_DIR / "Emotion_little_vgg.h5"
CLASS_INDICES_PATH = BASE_DIR / "class_indices.json"


def build_emotion_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", input_shape=(48, 48, 1)))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, kernel_initializer="he_normal"))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(64, kernel_initializer="he_normal"))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(5, kernel_initializer="he_normal"))
    model.add(Activation("softmax"))

    return model


class EmotionEngine:
    def __init__(self):
        if not CASCADE_PATH.exists():
            raise FileNotFoundError(f"Cascade file not found: {CASCADE_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        self.face_classifier = cv2.CascadeClassifier(str(CASCADE_PATH))
        self.classifier = self._load_model()
        self.class_labels = self._load_labels()

    def _load_model(self):
        try:
            return load_model(str(MODEL_PATH), compile=False)
        except Exception:
            model = build_emotion_model()
            model.load_weights(str(MODEL_PATH))
            return model

    def _load_labels(self):
        if CLASS_INDICES_PATH.exists():
            with CLASS_INDICES_PATH.open("r", encoding="utf-8") as file:
                class_indices = json.load(file)
            return [label for label, _ in sorted(class_indices.items(), key=lambda item: item[1])]
        return ["Angry", "Fear", "Happy", "Sad", "Suprise"]

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        faces = self.face_classifier.detectMultiScale(equalized, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            faces = self.face_classifier.detectMultiScale(equalized, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))

        if len(faces) == 0:
            faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

        results = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) == 0:
                continue

            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = self.classifier.predict(roi, verbose=0)[0]
            label_index = int(np.argmax(preds))
            label = self.class_labels[label_index]
            confidence = float(preds[label_index])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            results.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "all_predictions": {
                        self.class_labels[i]: float(preds[i]) for i in range(len(self.class_labels))
                    },
                }
            )

        return frame, results