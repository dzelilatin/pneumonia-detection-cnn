import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETUP PATHS
home_folder = os.path.expanduser("~")
base_dir = os.path.join(home_folder, 'Downloads', 'chest_xray')
test_dir = os.path.join(base_dir, 'test')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. LOAD DATA & MODEL
print("Loading Model & Data...")
model = tf.keras.models.load_model('pneumonia_model_v1.keras')

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False # Important for evaluation order
)

# 3. GENERATE PREDICTIONS
print("Predicting...")
predictions = model.predict(test_ds)
predicted_classes = (predictions > 0.5).astype("int32")
# Extract true labels
true_classes = np.concatenate([y for x, y in test_ds], axis=0)

# 4. GENERATE FIGURE 4.1: TRAINING HISTORY PLOT
# (Simulating the plot based on your logs since we didn't save the history object)
epochs = [1, 2, 3, 4, 5]
acc = [0.7198, 0.7910, 0.8369, 0.8732, 0.8942]
loss = [0.5645, 0.4309, 0.3543, 0.3088, 0.2729]
val_acc = [0.5000, 0.5625, 0.5625, 0.6250, 0.6875]

plt.figure(figsize=(10, 5))
plt.plot(epochs, acc, label='Training Accuracy', marker='o')
plt.plot(epochs, loss, label='Training Loss', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='x', linestyle='--')
plt.title('Training Dynamics: Accuracy & Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show() 
print("^^^ SCREENSHOT THIS FOR FIGURE 4.1 ^^^")

# 5. GENERATE FIGURE 4.2 & TABLE DATA
print("\n--- COPY THESE NUMBERS FOR TABLE 4.1 ---")
print(classification_report(true_classes, predicted_classes, target_names=['Normal', 'Pneumonia']))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
print("^^^ SCREENSHOT THIS FOR FIGURE 4.2 ^^^")