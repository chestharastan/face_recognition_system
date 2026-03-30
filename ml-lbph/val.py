import os
import cv2

MODEL_PATH = "trainer.yml"
LABEL_PATH = "machineface/labels.txt"
VALIDATION_PATH = "split_dataset_c/validation"

def load_labels(label_file):
    label_map = {}
    with open(label_file, "r") as f:
        for line in f:
            id_, name = line.strip().split(",", 1)
            label_map[int(id_)] = name
    return label_map

def test_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    label_map = load_labels(LABEL_PATH)

    total = 0
    correct = 0

    for person_name in os.listdir(VALIDATION_PATH):
        person_folder = os.path.join(VALIDATION_PATH, person_name)

        if not os.path.isdir(person_folder):
            continue

        for file_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, file_name)

            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces = face_detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = img[y:y+h, x:x+w]
                predicted_id, confidence = recognizer.predict(face_roi)
                predicted_name = label_map.get(predicted_id, "Unknown")

                total += 1
                if predicted_name == person_name:
                    correct += 1

                print(f"Image: {image_path}")
                print(f"Actual: {person_name}")
                print(f"Predicted: {predicted_name}")
                print(f"Confidence: {confidence}")
                print("-" * 30)

    if total > 0:
        accuracy = (correct / total) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    else:
        print("No faces found in validation set.")

if __name__ == "__main__":
    test_model()