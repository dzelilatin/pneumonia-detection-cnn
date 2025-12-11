import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

layers = tf.keras.layers
models = tf.keras.models
callbacks = tf.keras.callbacks

# --- CONFIGURATION ---
BATCH_SIZE = 32
IMG_HEIGHT = 180 
IMG_WIDTH = 180
EPOCHS = 20      # We use Early Stopping, so we can set this high
DATA_DIR = '/Users/user/Desktop/ml-project/chest_xray/train' 

# --- 1. LOAD DATA ---
train_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(f"Classes found: {class_names}")

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. DATA AUGMENTATION BLOCK (Fixes Overfitting) ---
data_augmentation = models.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
])

# --- 3. MODEL ARCHITECTURE (With Dropout) ---
model = models.Sequential([
  # Input & Rescaling
  layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  data_augmentation, # <--- Augmentation applied here
  layers.Rescaling(1./255),
  
  # Conv Block 1
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Conv Block 2
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Conv Block 3
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2), # <--- Dropout to reduce overfitting

  # Dense Layers
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.5), # <--- Higher Dropout here is crucial
  layers.Dense(len(class_names), activation='softmax') # or 'sigmoid' if binary
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# --- 4. TRAINING WITH EARLY STOPPING ---
# Stops training if validation loss doesn't improve for 3 epochs
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS,
  callbacks=[early_stopping]
)

# --- 5. PLOT TRAINING RESULTS ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('Figure_Final_Dynamics.png')
plt.show()

# --- 6. GENERATE CONFUSION MATRIX & METRICS ---
print("\nGenerating Evaluation Metrics...")

# Get predictions for the entire validation set
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
os.makedirs('results', exist_ok=True)
plt.savefig('results/Figure_Confusion_Matrix.png')
plt.show()

# Classification Report (Precision, Recall, F1)
report = classification_report(y_true, y_pred, target_names=class_names)
print("\n--- CLASSIFICATION REPORT ---")
print(report)

# Save Model for Phase 2
model.save('my_best_model.keras')
print("Model saved as my_best_model.keras")