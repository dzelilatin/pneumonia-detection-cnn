# Pneumonia Detection from Chest X-Rays 🫁

A Convolutional Neural Network (CNN) trained to classify Pneumonia vs. Normal chest X-rays. This project achieves **96% accuracy** and includes **Grad-CAM interpretability** to visualize the model's decision-making process.

## 📊 Performance Results
- **Accuracy:** 96%
- **Precision:** 98%
- **Recall:** 97%
- **F1-Score:** 0.96

![Accuracy Plot](results/accuracy_plot.png)
*Figure 1: Training dynamics showing no overfitting.*

![Confusion Matrix](results/confusion_matrix.png)
*Figure 2: Confusion Matrix.*

## 🧠 Explainability (Grad-CAM)
We implemented **Grad-CAM** (Gradient-weighted Class Activation Mapping) to verify that the model looks at the lungs rather than background artifacts.

![Heatmap](results/heatmap_explanation.jpg)
*Figure 3: Grad-CAM heatmap showing the model focusing on lung opacity.*

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python3 train_improved.py`
4. Run visualization: `python3 explainability.py`

## 📂 Project Structure
- `train_improved.py`: The main CNN training script with Data Augmentation & Dropout.
- `explainability.py`: Generates Grad-CAM heatmaps.
- `saved_models/`: Contains the best trained model.
- `results/`: Output charts and visualizations.