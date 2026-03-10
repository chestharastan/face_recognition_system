import os
import cv2
import numpy as np

DATASET_PATH = "split_dataset_c/train"
MODEL_PATH = "machineface/trainer.yml"
LABELS_PATH = "machineface/labels.txt"

os.makedirs("machineface", exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_id = 0

for person in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_folder):
        continue

    label_map[current_id] = person

    for image in os.listdir(person_folder):
        img_path = os.path.join(person_folder, image)

        if not image.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # optional but recommended
        img = cv2.resize(img, (200, 200))

        faces.append(img)
        labels.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(labels))
recognizer.save(MODEL_PATH)

# Save labels
with open(LABELS_PATH, "w") as f:
    for id_, name in label_map.items():
        f.write(f"{id_},{name}\n")

print("Training complete")
print("Model saved to:", MODEL_PATH)
print("Labels saved to:", LABELS_PATH)
print(label_map)