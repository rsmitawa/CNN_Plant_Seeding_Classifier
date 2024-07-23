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

4. Run the main script:
   ```
   python src/main.py
   ```

## 🗃️ Dataset

- **Size**: 4,750 images
- **Dimensions**: 128x128 pixels (RGB)
- **Classes**: 12 distinct plant species
- **Distribution**: Imbalanced (221 to 654 samples per species)

## 🔧 Project Structure

```
plant-species-classification/
│
├── data/
│   ├── raw/                  # Raw image data
│   └── processed/            # Preprocessed image data
│
├── src/
│   ├── data_preprocessing.py # Data preprocessing scripts
│   ├── models.py             # CNN model architectures
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── tests/                    # Unit tests
│
├── requirements.txt          # Project dependencies
├── README.md
└── LICENSE
```

## 💻 Technical Approach

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
| Advanced | 73% | 0.70 |

## 🛠️ Technologies Used

- Python
- TensorFlow & Keras
- NumPy & Pandas
- Matplotlib & Seaborn
- OpenCV
- Scikit-learn

---
