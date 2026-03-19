
***

# 🧠 Sentiment Analysis Using EEG Signals

This project implementing a complete **machine learning and deep learning pipeline** for **sentiment/emotion classification** using EEG signals derived from the **DEAP** dataset. 

***

## 📌 Project Overview

This project predicts **emotional sentiment** (POSITIVE, NEGATIVE, NEUTRAL) from precomputed EEG features stored in `emotions.csv`. 
It benchmarks multiple **ML** and **DL** models on a common preprocessing pipeline to identify the most effective approach for EEG-based affective computing.

### Key Features

- Uses **DEAP (Database for Emotion Analysis using Physiological Signals)** EEG dataset. 
- Works on rich, engineered features: statistical, spectral (FFT), entropy, correlation, covariance, eigenvalues, and Riemannian geometry features. 
- Three sentiment classes based on **Valence** ratings: POSITIVE, NEGATIVE, NEUTRAL. 
- Clean preprocessing pipeline with NaN/Inf handling, variance filtering, scaling, and PCA (95% variance retained). 
- Trains and compares **7 models** (4 ML + 3 DL) under a unified evaluation setup. 

***

## 📊 Dataset Details

- **Source:** DEAP EEG emotion dataset (32 subjects, 40 trials each, 32-channel EEG at 512 Hz). [
- **Original labels:** Self-reported **Valence** and **Arousal** scores per trial.
- **This project’s labels (Valence-based):**
  - Valence > 5 → POSITIVE  
  - Valence < 5 → NEGATIVE  
  - Valence ≈ 5 → NEUTRAL  

### Feature Groups in `emotions.csv`

- `mean_*_a/b` – Mean amplitude in Alpha/Beta bands.
- `stddev_*` – Standard deviation (signal variability). 
- `moments_*` – Higher-order moments (skewness, kurtosis). 
- `max_*`, `min_*` – Amplitude extremes. 
- `fft_*` – FFT power spectrum (frequency-domain energy).
- `entropy*` – Signal entropy (complexity). 
- `correlate_*` – Inter-channel correlations. 
- `covmat_*` – Covariance matrices capturing spatial relationships. 
- `eigen_*` – Eigenvalues representing dominant patterns.
- `logm_*` – Log-matrix features (Riemannian geometry). 
> Suffix `_a` = Alpha band (8–13 Hz), `_b` = Beta band (13–30 Hz). 

***

## ⚙️ Methods and Models

### Preprocessing Pipeline

1. Replace `Inf` / `NaN` with column medians. 
2. Remove zero-variance features. 
3. Encode labels: NEGATIVE = 0, NEUTRAL = 1, POSITIVE = 2. 
4. Train/test split (80% / 20%, stratified).
5. Standardize features using **StandardScaler**.
6. Apply **PCA** to retain 95% variance (dimensionality reduction). 

### Machine Learning Models

- K-Nearest Neighbors (KNN).
- Support Vector Machine (SVM). 
- Random Forest. 
- XGBoost. 

### Deep Learning Models

- Multilayer Perceptron (MLP). 
- 1D Convolutional Neural Network (1D-CNN). 
- Long Short-Term Memory (LSTM). 

Each model is evaluated on the same train/test split to ensure **fair comparison**. 

***

## 📁 Repository Structure

Example structure (adapt to your actual repo):

```text
├── EEG_61.ipynb          # Main notebook: end-to-end pipeline
├── emotions.csv          # Preprocessed EEG features and labels (user upload)
├── band_boxplots.png     # Alpha/Beta band distribution visualization
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies (optional)
```

`EEG_61.ipynb` contains all steps: data loading, exploration, preprocessing, model training, evaluation, and visualizations (e.g., alpha vs beta band box plots).
***

## 🚀 How to Run

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

2. **Create and activate environment (optional but recommended)**

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

Create `requirements.txt` including typical libraries like:

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
tensorflow   # or torch, depending on your DL implementation
```

Then run:

```bash
pip install -r requirements.txt
```

4. **Open the notebook**

```bash
jupyter notebook EEG_61.ipynb
```

5. **Load the dataset**

When prompted in the notebook, upload your `emotions.csv` file using the file upload widget. 

6. **Run all cells**

Execute cells sequentially to reproduce preprocessing, training, and evaluation.

***

## 📈 Visualizations

The notebook generates plots such as:

- Box plots of **alpha vs beta band mean amplitudes** across POSITIVE, NEGATIVE, and NEUTRAL classes. 
- PCA variance and feature-space visualizations (if enabled). 

These help interpret how frequency bands and features relate to emotional states.

***

## 🧪 Evaluation

The models are compared using standard metrics such as:

- Accuracy on the held-out test set.  
- Confusion matrices per class.  
- (Optionally) Precision, recall, F1-score, and ROC curves.

You can easily extend the notebook to log additional metrics or cross-validation results.

***

## 🔮 Future Work

- Add cross-subject and subject-independent evaluation setups.  
- Experiment with raw EEG time-series using end-to-end deep models.  
- Explore additional emotion dimensions (Arousal, Dominance) beyond Valence.  
- Integrate model explainability (SHAP, feature importance) for EEG features.

***

## 📜 Acknowledgements

- **DEAP Dataset:** Koelstra et al., *DEAP: A Database for Emotion Analysis using Physiological Signals*, IEEE Transactions on Affective Computing, 2012. 
- This repository is for academic and research purposes only; please respect the original DEAP dataset license and citation requirements.

***
