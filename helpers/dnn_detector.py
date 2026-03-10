import cv2
import numpy as np

class DNNFaceDetector:

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            "source/deploy.prototxt",
            "source/res10_300x300_ssd_iter_140000.caffemodel"
        )

    def detect(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype("int")

                x = max(0, x)
                y = max(0, y)
                x1 = min(w, x1)
                y1 = min(h, y1)

                faces.append((x, y, x1 - x, y1 - y))

        return faces