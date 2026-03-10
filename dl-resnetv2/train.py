import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

from utils import save_class_names

# -------------------------------
# Configuration
# -------------------------------
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

TRAIN_DIR = "split_dataset_c/train"
VAL_DIR = "split_dataset_c/validation"

MODEL_SAVE_PATH = "deeplface/facenet_model7.h5"
LABELS_SAVE_PATH = "deeplface/class_names.txt"

# -------------------------------
# Data generators
# -------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

NUM_CLASSES = train_generator.num_classes

# Save label mapping
save_class_names(train_generator.class_indices, LABELS_SAVE_PATH)

print("Class mapping:", train_generator.class_indices)

# -------------------------------
# Build model
# -------------------------------
base_model = InceptionResNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Callbacks
# -------------------------------
callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# -------------------------------
# Train
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[callback]
)

# -------------------------------
# Save model
# -------------------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)

print(f"Model saved to {MODEL_SAVE_PATH}")
print(f"Labels saved to {LABELS_SAVE_PATH}")

# -------------------------------
# Plot training history
# -------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.show()

print(f"Highest training accuracy: {max(history.history['accuracy']):.4f}")
print(f"Highest validation accuracy: {max(history.history['val_accuracy']):.4f}")