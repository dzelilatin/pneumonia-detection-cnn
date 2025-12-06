import os
import tensorflow as tf
from keras import layers
from keras.applications import VGG16

# --- 1. SETUP PATHS ---
home_folder = os.path.expanduser("~")
base_dir = os.path.join(home_folder, 'Downloads', 'chest_xray')

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# --- 2. HYPERPARAMETERS ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001

print(f"✅ TF Version: {tf.__version__}")
print("Loading data using modern Keras API...")

# --- 3. DATA LOADING (The Modern Way) ---
# This replaces ImageDataGenerator. It is faster and built into TensorFlow.

# Load Training Data
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True
)

# Load Validation Data
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

# Optimize performance (Prefetching makes it run faster on Mac)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. DEFINE AUGMENTATION & PREPROCESSING ---
# In modern Keras, we add augmentation as Layers inside the model!
# This is much cleaner.

augmentation_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# --- 5. BUILD MODEL: TRANSFER LEARNING ---
print("Building VGG16 Model...")

# Load VGG16 base
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze weights

# Create the Final Model
inputs = tf.keras.Input(shape=(224, 224, 3))

# 1. Apply Augmentation (Only runs during training automatically)
x = augmentation_layer(inputs)

# 2. Rescale pixel values (0-255 -> 0-1)
x = layers.Rescaling(1./255)(x)

# 3. Pass through VGG16
x = base_model(x, training=False)

# 4. Classification Head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
)

model.summary()

# --- 6. TRAIN ---
print("\nSTARTING TRAINING...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

# --- 7. SAVE ---
print("\nSaving model...")
model.save('pneumonia_model_v1.keras') # .keras is the new standard file format
print("✅ Model saved as 'pneumonia_model_v1.keras'")