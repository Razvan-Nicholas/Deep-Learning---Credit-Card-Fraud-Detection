# Deep-Learning---Credit-Card-Fraud-Detection
This repository serves the purpose of showcasing a Deep Learning project about Credit Card Fraud Detection using Deep Learning
This repository contains a Deep Learning project for credit card fraud detection. The project follows a research-paper-style methodology and compares a **Multi-Layer Perceptron (MLP)** against traditional machine learning models on a highly imbalanced real-world dataset.

---

## 1. Repository Structure

```
credit-card-fraud-detection/
│
├── notebooks/
│   └── fraud_detection_experiment.ipynb
│
├── src/
│   ├── data_utils.py
│   ├── models_ml.py
│   ├── models_dl.py
│   └── evaluation.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── requirements.txt
├── README.md
```

---

## 2. Environment Setup

### 2.1 Prerequisites
- Python **3.9+**
- pip or conda

### 2.2 Create Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

### 2.3 Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**
```txt
numpy
pandas
scikit-learn
matplotlib
seaborn
imbalanced-learn
tensorflow
jupyter
```

---

## 3. Dataset Setup

1. Download the dataset from Kaggle:
   *Credit Card Fraud Detection*  
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Place `creditcard.csv` inside:
```
credit-card-fraud-detection/data/
```

---

## 4. Source Code

### 4.1 Data Utilities (`src/data_utils.py`)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)


def split_and_scale(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
```

---

### 4.2 Traditional ML Models (`src/models_ml.py`)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def logistic_regression():
    return LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        random_state=42
    )


def random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
```

---

### 4.3 Deep Learning Model (`src/models_dl.py`)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_mlp(input_dim, layers=[32, 16], lr=0.001):
    model = Sequential()
    for i, units in enumerate(layers):
        if i == 0:
            model.add(Dense(units, activation='relu', input_dim=input_dim))
        else:
            model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['Precision', 'Recall']
    )
    return model
```

---

### 4.4 Evaluation (`src/evaluation.py`)

```python
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test, is_dl=False):
    if is_dl:
        y_pred = (model.predict(X_test) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print('ROC-AUC:', roc_auc_score(y_test, y_pred))
```

---

## 5. Main Experiment Notebook

### `notebooks/fraud_detection_experiment.ipynb`

**Workflow:**
1. Load dataset
2. Preprocess and scale features
3. Apply SMOTE (training set only)
4. Train Logistic Regression, Random Forest, MLP
5. Evaluate using precision, recall, F1-score, ROC-AUC
6. Save figures and tables

*(All hyperparameters and random seeds are fixed for reproducibility.)*

---

## 6. Reproducing Results

```bash
jupyter notebook notebooks/fraud_detection_experiment.ipynb
```

Run all cells sequentially.

Results (tables and figures) will be saved automatically in:
```
results/
```

---

## 7. Experimental Rigor

- Stratified train/test split
- No data leakage (SMOTE applied only on training data)
- Fixed random seeds
- Multiple evaluation metrics suitable for imbalanced learning

---

## 8. References

- Dal Pozzolo et al., *Calibrating Probability with Undersampling*, IEEE, 2015
- Kaggle Credit Card Fraud Dataset
- Medium: *Build a Model for Credit Card Fraud Detection*
- Goodfellow et al., *Deep Learning*, MIT Press

