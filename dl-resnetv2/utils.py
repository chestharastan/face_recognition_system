import os

def save_class_names(class_indices, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sorted_classes = sorted(class_indices.items(), key=lambda x: x[1])

    with open(save_path, "w") as f:
        for class_name, class_id in sorted_classes:
            f.write(f"{class_id},{class_name}\n")


def load_class_names(path):
    class_map = {}
    with open(path, "r") as f:
        for line in f:
            class_id, class_name = line.strip().split(",", 1)
            class_map[int(class_id)] = class_name
    return class_map