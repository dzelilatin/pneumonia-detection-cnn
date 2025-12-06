import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random

# --- CONFIGURATION ---
MODEL_PATH = 'saved_models/my_best_model.keras'
# We only point to the FOLDER now, not a specific file
TEST_FOLDER = '/Users/user/Desktop/ml-project/chest_xray/test/PNEUMONIA' 
IMG_SIZE = (180, 180)

# Shortcuts
layers = tf.keras.layers
models = tf.keras.models

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def build_functional_model():
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# --- MAIN EXECUTION ---
print("1. Loading model...")
old_model = models.load_model(MODEL_PATH)
clean_model = build_functional_model()
clean_model.set_weights(old_model.get_weights())

# Determine the last conv layer
clean_model.predict(np.zeros((1, 180, 180, 3)), verbose=0) # Warmup
last_conv_layer_name = ""
for layer in reversed(clean_model.layers):
    if 'conv2d' in layer.name:
        last_conv_layer_name = layer.name
        break

# --- FIND IMAGES ---
print("2. Hunting for the best images...")
all_files = os.listdir(TEST_FOLDER)
# Filter for "bacteria" images only (they usually look better)
bacteria_files = [f for f in all_files if "bacteria" in f and f.endswith(".jpeg")]
# Pick 10 random ones to test
selected_files = random.sample(bacteria_files, min(10, len(bacteria_files)))

print(f"   Found {len(selected_files)} images to process.")

os.makedirs('results', exist_ok=True)

for i, filename in enumerate(selected_files):
    img_path = os.path.join(TEST_FOLDER, filename)
    print(f"   Processing {i+1}/10: {filename}...")
    
    # Load & Preprocess
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_array, clean_model, last_conv_layer_name)
    
    # Style it "Journal Quality"
    img_cv2 = cv2.imread(img_path)
    img_cv2 = cv2.resize(img_cv2, IMG_SIZE)
    
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Transparency Trick:
    # High activation = Opaque. Low activation = Transparent.
    weights = heatmap_resized[:, :, np.newaxis] 
    
    # We boost the intensity (weights * 0.6) to make it "Glow" more
    superimposed_img = (heatmap_colored * weights * 0.6) + (img_cv2 * (1 - (weights * 0.6)))
    
    save_name = f'results/heatmap_{i+1}_{filename}'
    cv2.imwrite(save_name, superimposed_img)

print("\nDONE! Go look in your 'results' folder.")
print("Pick the best looking image and rename it to 'final_heatmap.jpg'!")