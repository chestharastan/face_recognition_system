import sys
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from helpers.haar_detector import HaarFaceDetector
from helpers.dnn_detector import DNNFaceDetector

MODEL_PATH = os.path.join(BASE_DIR, "deeplface", "facenet_model7.h5")
LABEL_PATH = os.path.join(BASE_DIR, "deeplface", "class_names.txt")
IMAGE_SIZE = (160, 160)
CONFIDENCE_THRESHOLD = 0.7


class CustomScaleLayer(Layer):
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 2:
                raise ValueError(f"CustomScaleLayer expected 2 inputs, got {len(inputs)}")
            x, shortcut = inputs
            return shortcut + x * self.scale
        return inputs * self.scale

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            return input_shape[0]
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


def load_labels(path):
    label_map = {}
    with open(path, "r") as f:
        for line in f:
            idx, name = line.strip().split(",", 1)
            label_map[int(idx)] = name
    return label_map


print("Loading labels...")
labels = load_labels(LABEL_PATH)
print("Labels loaded.")

print("Loading model...")
model = load_model(
    MODEL_PATH,
    custom_objects={"CustomScaleLayer": CustomScaleLayer},
    compile=False
)
print("Model loaded.")

print("Loading detector...")
# detector = HaarFaceDetector()
detector = DNNFaceDetector()
print("Detector loaded.")

print("Opening camera...")
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise IOError("Cannot open webcam")

print("Camera started. Press q to quit.")

try:
    while True:
        ret, frame = cam.read()

        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = detector.detect(gray)
        faces = detector.detect(frame)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = cv2.resize(face, IMAGE_SIZE)
            face = face.astype("float32")
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            predictions = model.predict(face, verbose=0)
            predicted_id = int(np.argmax(predictions))
            confidence = float(np.max(predictions))

            if confidence > CONFIDENCE_THRESHOLD:
                name = labels.get(predicted_id, "Unknown")
                text = f"{name} ({confidence:.2f})"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.imshow("DL Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam.release()
    cv2.destroyAllWindows()