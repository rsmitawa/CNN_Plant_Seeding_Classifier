--- 

# 🌿 Advanced Plant Species Classification using CNNs

## 📊 Project Overview

This project implements an advanced plant species classification system using Convolutional Neural Networks (CNNs). It accurately identifies 12 different plant species from image data, with potential applications in agriculture, botany, and environmental monitoring.

## 🚀 Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/rsmitawa/CNN_Plant_Seeding_Classifier.git
   cd CNN_Plant_Seeding_Classifier
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## 🗃️ Dataset

- **Size**: 4,750 images
- **Dimensions**: 128x128 pixels (RGB)
- **Classes**: 12 distinct plant species
- **Distribution**: Imbalanced (221 to 654 samples per species)

## 💻 Technical Approach

### 🛠️ Technologies Used

- TensorFlow
- Keras
- OpenCV
- Pillow - Python Imaging Library
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn

### Data Preprocessing

1. Gaussian Blurring: Reduces image noise
2. Normalization: Scales pixel values to [0, 1] range

### Model Architectures

We implemented two CNN architectures:

1. **Baseline Model**: 3 sets of Convolutional + Pooling layers
2. **Advanced Model**: Deeper architecture with optimized hyperparameters

### Training

- Optimizer: Adam (lr=0.001, beta_1=0.9, beta_2=0.999)
- Loss Function: Categorical Crossentropy
- Early Stopping and Model Checkpointing implemented

## 🚀 Results

| Model | Accuracy | Weighted F1-Score |
|-------|----------|-------------------|
| Baseline | 47% | 0.41 |
| Complex CNN | 73% | 0.70 |

## Conclusion and key takeaways

- The Convnet layer in Model 2 made the model much better and reduced the number of weights it needed to learn.
- Model 2 has a good F1-score, with a big drop in False Negatives and a small rise in False Positives.
- Both models did poorly for class 0, so a closer look at this class is needed.
- The final model has an accuracy of 73% and a weighted F1-score of 70%.
- Model 2 performs best on classes 10, 6, and 3, correctly recalling over 90% of them on the test set.

---
